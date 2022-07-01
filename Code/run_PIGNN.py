import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

import os
from itertools import chain, islice

from utils import *


import pyvista

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

from torch_scatter import scatter

node_types = {
    "field": 0,
    "bbox": 1,
    "farfield": 2,
    "airfoil": 3,
    "airfoil_edge": 4,
    "velocity_inlet": 5,
    "pressure_inlet": 6,
    "velocity_outlet": 7,
    "pressure_outlet": 8,
    "euler_wall": 9,
}

su2_to_node_type = {
    0: node_types["field"],  # volume field
    1: node_types["euler_wall"],  # airfoil
    2: node_types["euler_wall"],  # lower wall
    3: node_types["velocity_inlet"],  # inlet
    4: node_types["pressure_outlet"],  # outlet
    5: node_types["euler_wall"],  # upper wall
}


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

        h0 = self.compute_initial_latents(
            data.node_type_ids, data.node_lvls, data.u_bc
        )  # .view(1, 7421, 16)
        # h0 = self.compute_initial_latents(node_type_ids, node_lvls, u_bc, u_x_bc, u_xx_bc)

        # print(h0.shape)

        h1 = self.propagate(data.edge_index, h=h0, x=data.x) + h0
        h2 = self.propagate(data.edge_index, h=h1, x=data.x) + h1
        h3 = self.propagate(data.edge_index, h=h2, x=data.x) + h2
        h4 = self.propagate(data.edge_index, h=h1, x=data.x) + h3
        h5 = self.propagate(data.edge_index, h=h2, x=data.x) + h4
        h6 = self.propagate(data.edge_index, h=h1, x=data.x) + h5
        h7 = self.propagate(data.edge_index, h=h2, x=data.x) + h6

        # x_out=x_samples,
        # simplex_indices=simplex_indices,
        # simplex_transforms=[torch.tensor(tri.transform) for tri in tris]

        h_interp = self.interpolate_latents(
            data.x_out,
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


if __name__ == "__main__":

    print("Starting")

    plt.style.use("ggplot")

    print("Loading Simulation Data")

    data_dir = "../Data/Inc_Inviscid_Hydrofoil"
    mesh_filename = "mesh_NACA0012_5deg_6814.su2"

    mesh = read_su2_mesh(os.path.join(data_dir, mesh_filename))

    ### warning: some nodes might have multiple su2 tags (because originally only cells have su2 tags)

    node_type_ids = np.empty(len(mesh.points), dtype=int)
    node_su2_tags = np.empty(len(mesh.points), dtype=int)

    for i, su2_tags in enumerate(mesh.cell_data["su2:tag"]):
        for su2_tag in np.unique(su2_tags):
            cell_inds = np.where(su2_tags == su2_tag)[0]
            node_inds = mesh.cells[i].data[cell_inds].reshape(-1)
            node_type_ids[node_inds] = su2_to_node_type[su2_tag]
            node_su2_tags[node_inds] = su2_tag

    node_type_ids = node_type_ids.tolist()

    farfield_node_indices = np.where(np.array(node_su2_tags) >= 2)[0]
    airfoil_node_indices = np.where(np.array(node_su2_tags) == 1)[0]

    bbox_node_indices = [
        farfield_node_indices[
            np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[1], [1]]))
        ],
        farfield_node_indices[
            np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[1], [-1]]))
        ],
        farfield_node_indices[
            np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[-1], [1]]))
        ],
        farfield_node_indices[
            np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[-1], [-1]]))
        ],
    ]

    airfoil_edge_node_indices = [
        airfoil_node_indices[
            np.argmax(mesh.points[airfoil_node_indices, :2] @ np.array([[1], [0]]))
        ],
        airfoil_node_indices[
            np.argmax(mesh.points[airfoil_node_indices, :2] @ np.array([[-1], [0]]))
        ],
    ]

    node_pts = []
    node_orig_ids = []
    node_new_ids = []
    # node_type_ids = []
    node_lvls = []
    edges = []
    # keep_node_during_sampling = []
    node_subsampling_prob = []
    tris = []
    cells = []

    for i in range(len(mesh.points)):
        node_pts += [[mesh.points[i, 0], mesh.points[i, 1]]]
        node_orig_ids += [i]
        node_new_ids += [i]
        # node_type_ids += [0]
        node_lvls += [0]
        node_subsampling_prob += [0.5]

    node_type_ids = np.empty(len(mesh.points), dtype=int)

    for i, su2_tags in enumerate(mesh.cell_data["su2:tag"]):
        for su2_tag in np.unique(su2_tags):
            cell_inds = np.where(su2_tags == su2_tag)[0]
            node_inds = mesh.cells[i].data[cell_inds].reshape(-1)
            node_type_ids[node_inds] = su2_to_node_type[su2_tag]

    node_type_ids = node_type_ids.tolist()

    for i in farfield_node_indices:
        #     node_type_ids[i] = node_types["farfield"]
        node_subsampling_prob[i] = 0.75

    for i in bbox_node_indices:
        node_type_ids[i] = node_types["bbox"]
        # print("bbox", i, node_type_ids[i])
        node_subsampling_prob[i] = 1

    for i in airfoil_node_indices:
        #     node_type_ids[i] = node_types["airfoil"]
        node_subsampling_prob[i] = 0.75

    for i in airfoil_edge_node_indices:
        # node_type_ids[i] = node_types["airfoil_edge"]
        node_subsampling_prob[i] = 1

    print("Creating Multi-level Graph")

    print(0, len(node_pts))

    nodes_lvl = np.where(np.array(node_lvls) == 0)[0]
    node_pts_lvl = np.array(node_pts)[nodes_lvl]

    tri = Delaunay(node_pts_lvl)
    tris.append(tri)

    cells.append(tri.simplices + nodes_lvl.min())

    indptr, indices = tri.vertex_neighbor_vertices
    for k in range(len(indptr) - 1):
        i = indptr[k]
        j = indptr[k + 1]
        for m in indices[i:j]:
            edge = [nodes_lvl[k], nodes_lvl[m]]
            edges.append(edge)

    for lvl in range(1, 6):

        nodes_lvl = np.where(np.array(node_lvls) == lvl - 1)[0]
        p = np.random.rand(len(nodes_lvl))
        nodes_lvl = [
            n for i, n in enumerate(nodes_lvl) if node_subsampling_prob[n] >= p[i]
        ]
        # sample_indices = np.where(node_subsampling_prob[nodes_lvl.astype(np.int)] >= p)[0]
        # nodes_lvl = nodes_lvl[sample_indices]
        # nodes_lvl = nodes_lvl[node_subsampling_prob[nodes_lvl] >= p]

        for n in nodes_lvl:
            new_node_id = len(node_new_ids)
            node_pts += [node_pts[n]]
            node_orig_ids += [node_orig_ids[n]]
            node_new_ids += [new_node_id]
            node_type_ids += [node_type_ids[n]]
            node_lvls += [lvl]
            node_subsampling_prob += [node_subsampling_prob[n]]
            edges += [[new_node_id, n], [n, new_node_id]]

        print(lvl, len(node_pts))

        # create edges by Delauney triangulation
        nodes_lvl = np.where(np.array(node_lvls) == lvl)[0]
        node_pts_lvl = np.array(node_pts)[nodes_lvl]

        tri = Delaunay(node_pts_lvl)
        tris.append(tri)

        cells.append(tri.simplices + nodes_lvl.min())

        indptr, indices = tri.vertex_neighbor_vertices
        for k in range(len(indptr) - 1):
            i = indptr[k]
            j = indptr[k + 1]
            for m in indices[i:j]:
                edge = [nodes_lvl[k], nodes_lvl[m]]
                edges.append(edge)

    node_pts = np.array(node_pts)
    node_orig_ids = np.array(node_orig_ids)
    node_new_ids = np.array(node_new_ids)
    node_type_ids = np.array(node_type_ids)
    node_lvls = np.array(node_lvls)
    edges = np.array(edges)
    node_subsampling_prob = np.array(node_subsampling_prob)

    print("Preparing data")

    file_name = "flow.vtu"
    flow = pyvista.read(os.path.join(data_dir, file_name))

    data_field_names = [
        "Pressure",
        "Velocity_x",
        "Velocity_y",
        "Pressure_Coefficient",
        "Density",
    ]

    # qois = [torch.tensor(flow.point_data[qoi]).reshape(13937, -1) for qoi in data_field_names]
    # qois = torch.cat(qois, axis=1)

    qois = torch.tensor(
        np.stack(
            (
                flow.point_data["Pressure"],
                flow.point_data["Velocity"][:, 0],
                flow.point_data["Velocity"][:, 1],
                flow.point_data["Pressure_Coefficient"],
                # flow.point_data["Density"],
            )
        )
    ).T

    qois_mean = torch.mean(qois, dim=0)
    # qois_mean
    qois_std = torch.std(qois, dim=0)
    # qois_std

    qois_scaled = (qois - qois_mean) / qois_std

    u_bc = torch.zeros((node_pts.shape[0], qois.shape[1]))

    # INC_DENSITY_INIT= 998.2

    # u_bc[:,4] = INC_DENSITY_INIT

    nodes_field = np.where(node_type_ids == 0)[0]

    nodes_velocity_inlet = np.where(node_type_ids == 0)[0]
    u_bc[nodes_velocity_inlet, 1] = 1.775  # x-velocity

    nodes_pressure_outlet = np.where(node_type_ids == 0)[0]
    u_bc[nodes_pressure_outlet, 0] = 0.0  # pressure

    nodes_euler_wall = np.where(node_type_ids == 0)[0]

    tri0 = tris[0]

    n_samples = 10_000
    i_samples = np.random.choice(len(tri0.vertices), n_samples, replace=True)
    w_samples = np.random.rand(n_samples, 3)
    w_samples = w_samples / w_samples.sum(axis=1)[:, np.newaxis]

    x_samples = np.sum(
        node_pts[tri0.vertices[i_samples]] * w_samples[:, :, np.newaxis], axis=1
    )

    x_samples = node_pts[np.where(node_lvls == 0)[0]]

    simplex_indices = torch.empty((len(x_samples), len(tris)), dtype=torch.long)
    simplex_node_ids = []
    for lvl, tri in enumerate(tris):
        i_min = np.min(np.where(node_lvls == lvl)[0])
        simplex_indices[:, lvl] = torch.tensor(
            tri.find_simplex(x_samples), dtype=torch.long
        )  # + i_min
        simplex_node_ids.append(torch.tensor(tri.vertices, dtype=torch.long) + i_min)

    edge_index = torch.tensor(edges, dtype=torch.long)

    # x, edge_index, cells, node_type_ids, node_lvls, u_bc, u_x_bc=None, u_xx_bc=None, x_out=None, tri_indices=None

    data = Data(
        x=torch.tensor(node_pts),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        # cells=cells,
        node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
        node_lvls=torch.tensor(node_lvls, dtype=torch.long),
        # pos=torch.tensor(node_pts),
        # y=qois_scaled[node_orig_ids].to(torch.float),
        y=qois_scaled.to(torch.float),
        u_bc=u_bc.to(torch.float),
        u_x_bc=None,
        u_xx_bc=None,
        u_t_bc=None,
        x_out=torch.tensor(x_samples),
        simplex_indices=simplex_indices,
        # simplex_transforms=[torch.tensor(tri.transform).to(device) for tri in tris],
        simplex_transforms=[torch.tensor(tri.transform) for tri in tris],
        simplex_node_ids=simplex_node_ids,
    )

    print("Creating PIGNN_Euler model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PIGNN_Euler(device=device).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    loss_hist = []

    print("Starting Training")

    model.train()
    for epoch in range(100_000):
        optimizer.zero_grad()
        out = model(data)
        loss = torch.nn.functional.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.detach())

        if epoch % 100 == 0:
            loss_hist = torch.tensor(loss_hist)
            print(epoch, loss.detach(), torch.mean(loss_hist), torch.std(loss_hist))
            loss_hist = []

    s = "pignn_model_"
    existing_models = [fn[12:15] for fn in os.listdir() if fn.startswith(s)]
    if existing_models:
        new_model_name = (
            s + "{:03d}".format(int(sorted(existing_models)[-1]) + 1) + ".pt"
        )
    else:
        new_model_name = s + "000" + ".pt"

    print("Saving model as", new_model_name)
    torch.save(model.state_dict(), new_model_name)
