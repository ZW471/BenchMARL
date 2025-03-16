import torch
import torch_scatter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


def localize(v, frames):
    if len(v.shape) == 2:
        v = v.unsqueeze(1)
    return torch.bmm(v, frames).squeeze(1)

def scalarization(v_s, frames):

    v_s = localize(v_s.transpose(-1, -2), frames).transpose(-1, -2)
    v_s = v_s.reshape(-1, 4)

    return v_s

def flatten(v):
    return v.reshape(-1, v.shape[-1] * 2)

def recover(v):
    return v.reshape(-1, 2, v.shape[-1] // 2)

class GCP(torch.nn.Module):
    def __init__(self, scaler_emb_dim, vector_emb_dim, scaler_in_dim=None, vector_in_dim=None, localized=False):
        super().__init__()
        self.scalar_emb_dim = scaler_emb_dim
        self.vector_emb_dim = vector_emb_dim
        self.scaler_in_dim = scaler_in_dim if scaler_in_dim is not None else self.scalar_emb_dim
        self.vector_in_dim = vector_in_dim if vector_in_dim is not None else self.vector_emb_dim
        self.activation = torch.nn.SiLU()
        self.localized = localized

        self.D_s = torch.nn.Sequential(
            torch.nn.Linear(self.vector_in_dim, 2),
            self.activation
        )
        self.D_z = torch.nn.Sequential(
            torch.nn.Linear(self.vector_in_dim, self.vector_emb_dim // 4),
            self.activation
        )
        self.U_z = torch.nn.Sequential(
            torch.nn.Linear(self.vector_emb_dim // 4, self.vector_emb_dim),
            self.activation
        )
        self.S_out = torch.nn.Sequential(
            torch.nn.Linear(
                self.scaler_in_dim + self.vector_emb_dim // 4 + 4,
                self.scalar_emb_dim
            ),
            self.activation
        )
        self.V_gate = torch.nn.Sequential(
            torch.nn.Linear(self.scalar_emb_dim, 1),
            torch.nn.Sigmoid()
        )


    def forward(self, s, v, frames):
        z = self.D_z(v)
        z_norm = torch.norm(z.transpose(-1, -2), dim=-1)

        v_s = self.D_s(v)
        if not self.localized:
            v_s = scalarization(v_s, frames)
        else:
            v_s = v_s.reshape(-1, 4)

        s = torch.cat([
            s,
            v_s,
            z_norm
        ], dim=-1)
        s = self.S_out(s)

        v_up = self.U_z(z)
        v = torch.einsum("bij, bk -> bij", v_up, self.V_gate(s))

        return s, v


class GCPMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="sum"):
        super().__init__(aggr=aggr)
        self.activation = torch.nn.SiLU()
        # Define embedding dimensions.
        scaler_emb_dim = out_channels
        vector_emb_dim = out_channels // 4
        scaler_in_dim = in_channels
        vector_in_dim = 2

        # Initialize modules.
        self.scaler_emb = torch.nn.Sequential(
            torch.nn.Linear(scaler_in_dim, scaler_emb_dim),
            self.activation
        )
        self.vector_emb = torch.nn.Sequential(
            torch.nn.Linear(vector_in_dim, vector_emb_dim),
            self.activation
        )
        self.gcp_fusion = GCP(scaler_emb_dim, vector_emb_dim,
                              scaler_emb_dim * 2, vector_emb_dim * 2 + 1, localized=True)
        self.scalar_att = torch.nn.Sequential(
            torch.nn.Linear(scaler_emb_dim, 1),
            torch.nn.Sigmoid()
        )
        self.vector_att = torch.nn.Sequential(
            torch.nn.Linear(vector_emb_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, s, v, pos, frames, edge_index):
        # Reshape vector features as in the original code.
        v = v.view(-1, v.shape[-1] // 2, 2).transpose(-1, -2)  # [num_nodes, 2, f]
        s = self.scaler_emb(s)
        v = self.vector_emb(v)
        # Flip edge_index so that first row becomes source and second row destination.
        s_update, v_update = self.propagate(edge_index, s=s, v=flatten(v), pos=pos, frames=flatten(frames))
        s = s + s_update
        return s

    def message(self, s_i, s_j, v_i, v_j, pos_i, pos_j, frames_i, frames_j):
        v_i = recover(v_i)
        v_j = recover(v_j)
        frames_i = recover(frames_i)
        frames_j = recover(frames_j)
        displacements = localize(pos_i - pos_j, frames_j).unsqueeze(1)

        # s_i: destination (target), s_j: source.
        # Build scalar message: concatenate source and destination scalars.
        x_ij = torch.cat([s_j, s_i], dim=-1)
        # For vector messages, localize both source and target features.
        v_i = (localize(v_i.transpose(-1, -2), torch.bmm(frames_i.transpose(-1, -2), frames_j)) + displacements).transpose(-1, -2)
        v_ij = torch.cat([v_j, v_i, displacements.transpose(-1, -2)], dim=-1)

        x_ij, v_ij = self.gcp_fusion(x_ij, v_ij, None)

        # Apply attention weighting.
        # x_ij = x_ij * self.scalar_att(x_ij)
        # v_ij = torch.einsum("bij, bmk -> bij", v_ij, self.vector_att(v_ij))
        v_ij = flatten(v_ij)
        return x_ij, v_ij

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Separate the two parts of the message.
        x, v = inputs
        x_aggr = scatter(x, index, dim=0, reduce=self.aggr, dim_size=dim_size)
        v_aggr = scatter(v, index, dim=0, reduce=self.aggr, dim_size=dim_size)
        return x_aggr, v_aggr

    def update(self, aggr_out):
        # The update step simply passes through the aggregated messages.
        return aggr_out


