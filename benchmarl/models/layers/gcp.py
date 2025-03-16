import torch
import torch_scatter
from torch_geometric.nn import MessagePassing

def localize(v, frames):
    if len(v.shape) == 2:
        v = v.unsqueeze(1)
    return torch.bmm(v, frames).squeeze(1)

def scalarization(v_s, frames):

    v_s = localize(v_s.transpose(-1, -2), frames).transpose(-1, -2)
    v_s = v_s.reshape(-1, 4)

    return v_s

class GCP(torch.nn.Module):
    def __init__(self, scaler_emb_dim, vector_emb_dim, scaler_in_dim=None, vector_in_dim=None):
        super().__init__()
        self.scalar_emb_dim = scaler_emb_dim
        self.vector_emb_dim = vector_emb_dim
        self.scaler_in_dim = scaler_in_dim if scaler_in_dim is not None else self.scalar_emb_dim
        self.vector_in_dim = vector_in_dim if vector_in_dim is not None else self.vector_emb_dim
        self.activation = torch.nn.SiLU()

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
        v_s = scalarization(v_s, frames)

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
        super().__init__(aggr)
        scaler_emb_dim = out_channels
        vector_emb_dim = out_channels
        scaler_in_dim = in_channels
        vector_in_dim = 2
        self.emb = GCP(scaler_emb_dim, vector_emb_dim, scaler_in_dim, vector_in_dim)
        self.gcp_fusion = GCP(scaler_emb_dim, vector_emb_dim, scaler_emb_dim * 2, vector_emb_dim * 2)
        self.aggr = aggr
        self.out = GCP(scaler_emb_dim, vector_emb_dim)


    def message_and_aggregate(self, x_i, x_j, v_i, v_j, _f, edge_index):
        f_i = _f[edge_index[1]]
        f_j = _f[edge_index[0]]
        x_ij = torch.cat([x_i, x_j], dim=-1)
        v_ij = torch.cat([
            localize(v_i.transpose(-1, -2), f_i.transpose(-1, -2)).transpose(-1, -2),
            localize(v_j.transpose(-1, -2), f_j.transpose(-1, -2)).transpose(-1, -2)
        ], dim=1)
        x_ij, v_ij = self.gcp_fusion(x_ij, v_ij, f_j)
        x_j = torch_scatter.scatter(x_ij, edge_index[0], dim=0, reduce=self.aggr, dim_size=x_j.shape[0])
        v_j = torch_scatter.scatter(v_ij, edge_index[0], dim=0, reduce=self.aggr, dim_size=v_j.shape[0])
        return torch.stack([x_j, v_j], dim=0)

    def update(self, aggr_out):
        return aggr_out[0], aggr_out[1]

    def forward(self, s, v, frames, edge_index):
        v = v.view(-1, v.shape[-1] // 2, 2).transpose(-1, -2)  # [b, 2, f]
        s, v = self.emb(s, v, frames)
        s_update, v_update = self.propagate(edge_index=edge_index, x=s, v=v, _f=frames)
        s, v = self.out(s + s_update, v + v_update, frames)
        return s #, localize(v_update + v, frames)

