import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

from torch_scatter import scatter


class PIGNN_Euler(MessagePassing):
    def __init__(
        self,
        device,
        space_dim=2,
        in_channels=4,
        latent_dim=32,
        out_channels=4,
        num_type_embeddings=10,
        type_embedding_dim=8,
        num_lvls=6,
        lvl_embedding_dim=4,
        delaunay_tris=None,
        num_residual_latent_updates=16,
    ):
        super().__init__(aggr="add", node_dim=0)  #  "Max" aggregation.

        self.latent_dim = latent_dim
        self.num_lvls = num_lvls
        self.device = device

        self.initial_latent_mlp = Seq(
            Linear(in_channels + type_embedding_dim + lvl_embedding_dim, 40),
            nn.Dropout(p=0.01),
            ReLU(),
            Linear(40, 40),
            ReLU(),
            Linear(40, latent_dim),
        )

        self.message_mlp = Seq(
            Linear(2 * latent_dim + 1 + space_dim, 40),
            nn.Dropout(p=0.01),
            ReLU(),
            Linear(40, 40),
            ReLU(),
            Linear(40, latent_dim),
        )

        self.output_mlp = Seq(
            Linear(latent_dim * num_lvls, 40),
            nn.Dropout(p=0.01),
            Sigmoid(),
            Linear(40, 40),
            Sigmoid(),
            Linear(40, 40),
            Sigmoid(),
            Linear(40, 40),
            Sigmoid(),
            Linear(40, out_channels),
        )

        # 0: farfield, 1: no slip boundary, 2: field, ...
        self.node_type_emb = nn.Embedding(num_type_embeddings, type_embedding_dim)
        self.node_lvl_emb = nn.Embedding(num_lvls, lvl_embedding_dim)

        self.delaunay_tris = delaunay_tris

        self.num_residual_latent_updates = num_residual_latent_updates

    # def forward(self, x, edge_index, cells, node_type_ids, node_lvls, u_bc, u_x_bc=None, u_xx_bc=None, x_out=None, tri_indices=None):

    #     h0 = self.compute_initial_latents(node_type_ids, node_lvls, u_bc)
    #     # h0 = self.compute_initial_latents(node_type_ids, node_lvls, u_bc, u_x_bc, u_xx_bc)

    #     h1 = self.propagate(edge_index, h=h0, x=x) + h0
    #     h2 = self.propagate(edge_index, h=h1, x=x) + h1
    #     h3 = self.propagate(edge_index, h=h2, x=x) + h2

    #     if x_out is None:
    #         u = self.output(h3)
    #     else:
    #         h_interp = self.interpolate_latents(x_out, x, h3, tri_indices)
    #         u = self.output(h_interp)

    #     return u

    # def forward(self, data):

    #     h0 = self.compute_initial_latents(data.node_type_ids, data.node_lvls, data.u_bc)    #.view(1, 7421, 16)
    #     # h0 = self.compute_initial_latents(node_type_ids, node_lvls, u_bc, u_x_bc, u_xx_bc)

    #     print(h0.shape)

    #     # edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))

    #     y = self.propagate(edge_index=data.edge_index, h=h0, x=data.x) #+ h0

    #     return y

    # def aggregate(self, inputs, index,
    #               ptr = None,
    #               dim_size = None):

    #     print("aggregate")
    #     print(inputs.shape)
    #     print(inputs)
    #     print(index.shape)
    #     print(index)
    #     print(ptr)
    #     print(dim_size)

    #     # out = torch.zeros((dim_size, inputs.shape[1]))

    #     # for i in index:
    #     #     out[i] += inputs[i]

    #     out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

    #     return out

    def forward(self, data):

        h7 = self.compute_graph_latents(data)

        # x_out=x_samples,
        # simplex_indices=simplex_indices,
        # simplex_transforms=[torch.tensor(tri.transform) for tri in tris]

        h_interp = self.interpolate_latents(
            data.x_data,
            h7,
            data.node_lvls,
            data.simplex_indices,
            data.simplex_transforms,
            data.simplex_node_ids,
        )

        # if data.x_out is None:
        # if "x_out" in data.keys:
        #     u = self.output(h3)
        # else:
        #     h_interp = self.interpolate_latents(data.x_out, data.x, h3, data.tri_indices)
        #     u = self.output(h_interp)

        # print("h1", h1)

        u = self.output(h_interp)

        return u

    def message(self, h_i, h_j, x_i, x_j):
        # print("message")
        # print(h_i, h_j, x_i, x_j)
        dx = x_j - x_i
        dist = torch.functional.norm(dx, p=2, dim=-1).view(-1, 1)
        # print(dx.shape, dist.shape)
        # n = dx / dist

        z_in = torch.cat((h_i, h_j - h_i, dist, dx), dim=-1).to(torch.float)
        z_out = self.message_mlp(z_in)
        # print("z_out", z_out)
        # print(z_out.shape)

        return z_out

    def compute_initial_latents(
        self, node_type_ids, node_lvls, u_bc, u_x_bc=None, u_xx_bc=None
    ):
        z_type = self.node_type_emb(node_type_ids)
        z_lvl = self.node_lvl_emb(node_lvls)

        # print(z_type.shape, z_lvl.shape, u_bc.shape)
        z0 = torch.cat((z_type, z_lvl, u_bc), dim=-1)
        # print(z0.shape)
        # z0 = torch.cat((z_type, z_lvl, u_bc, u_x_bc, u_xx_bc))
        h0 = self.initial_latent_mlp(z0)

        return h0

    # def compute_graph_latents(self, edge_index, x, node_type_ids, node_lvls, u_bc):
    #     pass

    def compute_graph_latents(self, data):
        h0 = self.compute_initial_latents(
            data.node_type_ids, data.node_lvls, data.u_bc
        )  # .view(1, 7421, 16)
        # h0 = self.compute_initial_latents(node_type_ids, node_lvls, u_bc, u_x_bc, u_xx_bc)

        # print(h0.shape)

        # h1 = self.propagate(data.edge_index, h=h0, x=data.x) + h0
        # h2 = self.propagate(data.edge_index, h=h1, x=data.x) + h1
        # h3 = self.propagate(data.edge_index, h=h2, x=data.x) + h2
        # h4 = self.propagate(data.edge_index, h=h1, x=data.x) + h3
        # h5 = self.propagate(data.edge_index, h=h2, x=data.x) + h4
        # h6 = self.propagate(data.edge_index, h=h1, x=data.x) + h5
        # h7 = self.propagate(data.edge_index, h=h2, x=data.x) + h6

        h_new = h0
        for i in range(self.num_residual_latent_updates):
            h_old = h_new
            h_new = self.propagate(data.edge_index, h=h_old, x=data.x) + h_old

        return h_new

    def interpolate_latents(
        self,
        x_out,
        h_nodes,
        node_lvls,
        simplex_indices,
        simplex_transforms,
        simplex_node_ids,
    ):

        lvls = torch.unique(node_lvls).tolist()
        h_interp = torch.zeros(
            (len(x_out), len(lvls), self.latent_dim), device=self.device
        )
        for lvl in lvls:
            r = simplex_transforms[lvl][simplex_indices[:, lvl], 2]
            # r.shape

            Tinv = simplex_transforms[lvl][simplex_indices[:, lvl], :2]
            # Tinv.shape

            c = (Tinv.transpose(0, 1) * (x_out - r)).sum(dim=2).T
            # c.shape

            w = torch.cat((c, 1 - c.sum(dim=1).view(-1, 1)), dim=1)
            # w.shape

            h_interp[:, lvl, :] = (
                w.view(-1, 3, 1)
                * h_nodes[simplex_node_ids[lvl][simplex_indices[:, lvl]]]
            ).sum(dim=1)

        h_interp = h_interp.reshape(-1, self.num_lvls * self.latent_dim)

        #     h_interp[:, lvl, :] = torch.sum(h_nodes[simplex_node_ids[lvl][simplex_indices]] * w, dim=-1)

        # h_interp = h_interp.reshape(len(x_out), -1)

        return h_interp

        # # source: https://stackoverflow.com/questions/57863618/how-to-vectorize-calculation-of-barycentric-coordinates-in-python

        # samples = x.reshape(-1, 2)
        # samples = np.vstack((x,0.5*x))

        # s = tri.find_simplex(samples)

        # b0 = (tri.transform[s, :2].transpose([1, 0, 2]) *
        #         (samples - tri.transform[s, 2])).sum(axis=2).T
        # coord = np.c_[b0, 1 - b0.sum(axis=1)]

    def output(self, h):
        # print(h)
        y = self.output_mlp(h)
        return y

    def residuals(self, data):

        u = self.forward(data)

        u_x = torch.autograd.grad(
            u,
            data.x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        # u_xx = torch.autograd.grad(u_x, data.x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        # u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        return
