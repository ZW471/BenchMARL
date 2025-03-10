from typing import Literal, Callable, Union

from torch import nn
import torch.nn.functional as F

ActivationType = Literal[
    "relu", "elu", "leaky_relu", "tanh", "sigmoid", "none", "silu", "swish"
]

def get_activations(
        act_name: ActivationType, return_functional: bool = False
) -> Union[nn.Module, Callable]:
    """Maps activation name (str) to activation function module."""
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    elif act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    elif act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    elif act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    elif act_name == "none":
        return nn.Identity()
    elif act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {act_name}")