#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import importlib
import inspect
import warnings
from dataclasses import dataclass, MISSING
from math import prod
from typing import List, Optional, Type

import torch
from tensordict import TensorDictBase
from tensordict.utils import _unravel_key_to_tuple, NestedKey
from torch import nn, Tensor

from .gnn import Gnn
from .common import Model, ModelConfig
from .gnn import _get_edge_index, _batch_from_dense_to_ptg

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None
if _has_torch_geometric:
    import torch_geometric
    from torch_geometric.transforms import BaseTransform

    class _RelVel(BaseTransform):
        """Transform that reads graph.vel and writes node1.vel - node2.vel in the edge attributes"""

        def __init__(self):
            pass

        def __call__(self, data):
            (row, col), vel, pseudo = data.edge_index, data.vel, data.edge_attr

            cart = vel[row] - vel[col]
            cart = cart.view(-1, 1) if cart.dim() == 1 else cart

            if pseudo is not None:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = cart
            return data


TOPOLOGY_TYPES = {"full", "empty", "from_pos"}


class GcpNet(Gnn):
    """A GNN model.

    GNN models can be used as "decentralized" actors or critics.

    Args:
        topology (str): Topology of the graph adjacency matrix. Options: "full", "empty", "from_pos". "from_pos" builds
            the topology dynamically based on ``position_key`` and ``edge_radius``.
        self_loops (str): Whether the resulting adjacency matrix will have self loops.
        gnn_class (Type[torch_geometric.nn.MessagePassing]): the gnn convolution class to use
        gnn_kwargs (dict, optional): the dict of arguments to pass to the gnn conv class
        position_key (str, optional): if provided, it will need to match a leaf key in the tensordict coming from the env
            (in the `observation_spec`) representing the agent position.
            To do this, your environment needs to have dictionary observations and one of the keys needs to be `position_key`.
            This key will be processed as a node feature (unless exclude_pos_from_node_features=True) and it will be used to construct edge features.
            In particular, it will be used to compute relative positions (``pos_node_1 - pos_node_2``) and a
            one-dimensional distance for all neighbours in the graph.
            If you want to use this feature in a :class:`~benchmarl.models.SequenceModel`, the GNN needs to be first in sequence.
        pos_features (int, optional): Needed when position_key is specified.
            It has to match to the last element of the shape the tensor under position_key.
        exclude_pos_from_node_features (optional, bool): If ``position_key`` is provided,
            wether to use it just to compute edge features or also include it in node features.
        velocity_key (str, optional): if provided, it will need to match a leaf key in the tensordict coming from the env
            (in the `observation_spec`) representing the agent position.
            To do this, your environment needs to have dictionary observations and one of the keys needs to be `velocity_key`.
            This key will be processed as a node feature, and
            it will be used to construct edge features. In particular, it will be used to compute relative velocities
            (``vel_node_1 - vel_node_2``) for all neighbours in the graph.
            If you want to use this feature in a :class:`~benchmarl.models.SequenceModel`, the GNN needs to be first in sequence.
        vel_features (int, optional): Needed when velocity_key is specified.
            It has to match to the last element of the shape the tensor under velocity_key.
        edge_radius (float, optional): If topology is ``"from_pos"`` the radius to use to build the agent graph.
            Agents within this radius distance will be neighnours.

    Examples:

        .. code-block:: python

            import torch_geometric
            from torch import nn
            from benchmarl.algorithms import IppoConfig
            from benchmarl.environments import VmasTask
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig
            experiment = Experiment(
                algorithm_config=IppoConfig.get_from_yaml(),
                model_config=GnnConfig(
                    topology="full",
                    self_loops=False,
                    gnn_class=torch_geometric.nn.conv.GATv2Conv,
                    gnn_kwargs={},
                ),
                critic_model_config=SequenceModelConfig(
                    model_configs=[
                        MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
                        GnnConfig(
                            topology="full",
                            self_loops=False,
                            gnn_class=torch_geometric.nn.conv.GraphConv,
                        ),
                        MlpConfig(num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear),
                    ],
                    intermediate_sizes=[5,3],
                ),
                seed=0,
                config=ExperimentConfig.get_from_yaml(),
                task=VmasTask.NAVIGATION.get_from_yaml(),
            )
            experiment.run()

    """

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = [
            tensordict.get(in_key)
            for in_key in self.in_keys
            if _unravel_key_to_tuple(in_key)[-1]
               not in (self.position_key, self.velocity_key)
        ]
        # Retrieve position
        if self.position_key is not None:
            if self._full_position_key is None:  # Run once to find full key
                self._full_position_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.position_key
                )
                pos = tensordict.get(self._full_position_key)
                if pos.shape[-1] != self.pos_features - 1:
                    raise ValueError(
                        f"Position key in tensordict is {pos.shape[-1]}-dimensional, "
                        f"while model was configured with pos_features={self.pos_features-1}"
                    )
            else:
                pos = tensordict.get(self._full_position_key)
            if not self.exclude_pos_from_node_features:
                input.append(pos)
        else:
            pos = None

        # Retrieve velocity
        if self.velocity_key is not None:
            if self._full_velocity_key is None:  # Run once to find full key
                self._full_velocity_key = self._get_key_terminating_with(
                    list(tensordict.keys(True, True)), self.velocity_key
                )
                vel = tensordict.get(self._full_velocity_key)
                if vel.shape[-1] != self.vel_features:
                    raise ValueError(
                        f"Velocity key in tensordict is {vel.shape[-1]}-dimensional, "
                        f"while model was configured with vel_features={self.vel_features}"
                    )
            else:
                vel = tensordict.get(self._full_velocity_key)
            input.append(vel)
        else:
            vel = None

        frames = tensordict.get(self._get_key_terminating_with(
            list(tensordict.keys(True, True)), "frames"
        )).view(-1, 2, 2)

        input = torch.cat(input, dim=-1)

        batch_size = input.shape[:-2]

        graph = _batch_from_dense_to_ptg(
            x=input,
            edge_index=self.edge_index,
            pos=pos,
            vel=vel,
            self_loops=self.self_loops,
            edge_radius=self.edge_radius,
        )
        forward_gnn_params = {
            "s": graph.x,
            "v": graph.vel,
            "frames": frames,
            "edge_index": graph.edge_index,
        }
        if (
                self.position_key is not None or self.velocity_key is not None
        ) and self.gnn_supports_edge_attrs:
            forward_gnn_params.update({"edge_attr": graph.edge_attr})

        if not self.share_params:
            if not self.centralised:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params).view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )[..., i, :]
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                )
            else:
                res = torch.stack(
                    [
                        gnn(**forward_gnn_params)
                        .view(
                            *batch_size,
                            self.n_agents,
                            self.output_features,
                        )
                        .mean(dim=-2)  # Mean pooling
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=-2,
                        )

        else:
            res = self.gnns[0](**forward_gnn_params).view(
                *batch_size, self.n_agents, self.output_features
            )
            if self.centralised:
                res = res.mean(dim=-2)  # Mean pooling

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class GcpNetConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Gnn`."""

    topology: str = MISSING
    self_loops: bool = MISSING

    gnn_class: Type[torch_geometric.nn.MessagePassing] = MISSING
    gnn_kwargs: Optional[dict] = None

    position_key: Optional[str] = None
    pos_features: Optional[int] = 0
    velocity_key: Optional[str] = None
    vel_features: Optional[int] = 0
    exclude_pos_from_node_features: Optional[bool] = None
    edge_radius: Optional[float] = None

    @staticmethod
    def associated_class():
        return GcpNet
