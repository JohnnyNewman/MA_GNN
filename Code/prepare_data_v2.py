import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import scipy.sparse.linalg
import scipy.spatial
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
    "heatflux_wall": 10,
}

su2_to_node_type = {
    0: node_types["field"],  # volume field
    1: node_types["heatflux_wall"],  # airfoil
    2: node_types["farfield"],  # farfield
}

# su2_to_node_type = {
#     0: node_types["field"],  # volume field
#     1: node_types["euler_wall"],  # airfoil
#     2: node_types["euler_wall"],  # lower wall
#     3: node_types["velocity_inlet"],  # inlet
#     4: node_types["pressure_outlet"],  # outlet
#     5: node_types["euler_wall"],  # upper wall
# }


def load_mesh(data_dir, mesh_filename):
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

    return mesh, node_su2_tags


def create_node_graph(mesh, node_su2_tags, num_levels=6, verbose=False):

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

    if verbose:
        print("Creating Multi-level Graph")

    if verbose:
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

    for lvl in range(1, num_levels):

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

        if verbose:
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

    return (
        node_pts,
        node_orig_ids,
        node_new_ids,
        node_type_ids,
        node_lvls,
        edges,
        node_subsampling_prob,
        tris,
        cells,
    )


def create_multilevel_graph(
    x0, node_type_ids, node_su2_tags, num_levels=8, verbose=False
):

    # num_levels = 8

    node_subsampling_prob = np.empty_like(x0[:, 0], dtype=np.float32)

    # volume
    node_subsampling_prob[node_su2_tags == 0] = 0.5

    # airfoil
    node_subsampling_prob[node_su2_tags == 1] = 0.75

    # farfield
    node_subsampling_prob[node_su2_tags == 2] = 0.75

    # bbox
    node_subsampling_prob[node_su2_tags == 6] = 1

    node_new_ids = np.arange(len(node_type_ids))
    nodes_last_lvl = node_new_ids
    node_orig_ids = node_new_ids
    node_lvls = np.zeros_like(node_new_ids)
    node_pts = x0
    tris = []

    tri = Delaunay(node_pts)
    tris.append(tri)

    cells = tri.simplices

    edges = []
    indptr, indices = tri.vertex_neighbor_vertices
    for k in range(len(indptr) - 1):
        i = indptr[k]
        j = indptr[k + 1]
        for m in indices[i:j]:
            edge = np.array([[node_new_ids[k], node_new_ids[m]]])
            edges.append(edge)

    if verbose:
        print(0, len(node_new_ids), len(node_new_ids), len(edges))

    for lvl in range(1, num_levels):
        p = np.random.rand(len(nodes_last_lvl))
        # nodes_lvl = [n for i,n in enumerate(nodes_last_lvl) if node_subsampling_prob[n] >= p[i]]
        nodes_lvl_sampled = nodes_last_lvl[node_subsampling_prob[nodes_last_lvl] >= p]
        node_orig_ids = np.concatenate((node_orig_ids, nodes_lvl_sampled))

        nodes_new_lvl = (
            np.arange(nodes_lvl_sampled.shape[0], dtype=node_new_ids.dtype)
            + node_new_ids.shape[0]
        )
        node_new_ids = np.concatenate((node_new_ids, nodes_new_lvl))

        np.stack((nodes_lvl_sampled, nodes_new_lvl), axis=1).shape
        edges.append(np.stack((nodes_lvl_sampled, nodes_new_lvl), axis=1))
        edges.append(np.stack((nodes_new_lvl, nodes_lvl_sampled), axis=1))

        # node_subsampling_prob_new = node_subsampling_prob[nodes_lvl]
        node_subsampling_prob = np.concatenate(
            (node_subsampling_prob, node_subsampling_prob[nodes_lvl_sampled])
        )

        node_lvls_new = np.ones_like(nodes_new_lvl) * lvl
        # node_lvls_new = lvl
        node_lvls = np.concatenate((node_lvls, node_lvls_new))

        node_pts_lvl = node_pts[nodes_lvl_sampled]
        node_pts = np.concatenate((node_pts, node_pts_lvl))

        node_new_type_ids = node_type_ids[nodes_lvl_sampled]
        node_type_ids = np.concatenate((node_type_ids, node_new_type_ids))
        node_su2_tags = np.concatenate(
            (node_su2_tags, node_su2_tags[nodes_lvl_sampled])
        )

        tri = Delaunay(node_pts_lvl)
        tris.append(tri)

        cells = np.concatenate((cells, tri.simplices + nodes_new_lvl.min()))

        indptr, indices = tri.vertex_neighbor_vertices
        for k in range(len(indptr) - 1):
            i = indptr[k]
            j = indptr[k + 1]
            for m in indices[i:j]:
                edge = np.array([[nodes_new_lvl[k], nodes_new_lvl[m]]])
                edges.append(edge)

        if verbose:
            print(lvl, len(nodes_lvl_sampled), len(node_new_ids), len(edges))

        nodes_last_lvl = nodes_new_lvl

    edges = np.concatenate(edges)

    return (
        node_pts,
        node_orig_ids,
        node_new_ids,
        node_type_ids,
        node_lvls,
        node_su2_tags,
        edges,
        node_subsampling_prob,
        tris,
        cells,
    )


def process_config(dir, fname, verbose=True):
    if verbose:
        print("Reading", os.path.join(dir, fname))
    with open(os.path.join(dir, fname)) as f:
        lines = f.readlines()
        # print(lines)
        # df = pd.DataFrame()
        df = {}
        for l in lines:
            # try:
            ls = l.split("=", 1)
            k = ls[0].strip()
            if k.startswith("%"):
                continue
            # v = ls[1].strip().split(", ")
            v = ls[1].strip()
            # print(k,v)
            # if k in ["MACH_NUMBER", "AOA", "REYNOLDS_NUMBER", "FREESTREAM_TEMPERATURE", "FREESTREAM_PRESSURE", "DV_VALUE"]:
            #     df[k] = pd.Series([v], dtype="str")
            df[k] = v
    return df


def load_sim_data_RANS(data_dir, file_name, node_type_ids, D=2):

    flow = pyvista.read(os.path.join(data_dir, file_name))

    # qois = [torch.tensor(flow.point_data[qoi]).reshape(13937, -1) for qoi in data_field_names]
    # qois = torch.cat(qois, axis=1)

    qois = torch.tensor(
        np.stack(
            (
                flow.point_data["Density"],
                flow.point_data["Pressure"],
                # flow.point_data["Velocity"][:, 0],
                # flow.point_data["Velocity"][:, 1],
                flow.point_data["Mach"],
                flow.point_data["Momentum"][:, 0],
                flow.point_data["Momentum"][:, 1],
                flow.point_data["Energy"],
                flow.point_data["Temperature"],
                flow.point_data["Laminar_Viscosity"],
                flow.point_data["Eddy_Viscosity"],
                flow.point_data["Nu_Tilde"],
                flow.point_data["Y_Plus"],
            )
        )
    ).T

    qoi_names = [
        "Density",
        "Pressure",
        # "Velocity_x",
        # "Velocity_y",
        "Mach",
        "Momentum_x",
        "Momentum_y",
        "Energy",
        "Temperature",
        "Laminar_Viscosity",
        "Eddy_Viscosity",
        "Nu_Tilde",
        "Y_Plus",
    ]

    qois_mean = torch.mean(qois, dim=0)
    # qois_mean
    qois_std = torch.std(qois, dim=0)
    # qois_std

    qois_scaled = (qois - qois_mean) / qois_std

    return (
        qoi_names,
        qois,
        qois_scaled,
        qois_mean,
        qois_std,
        flow.points[:, :D],
    )


def load_sim_data_euler(data_dir, file_name, node_type_ids, D=2):

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
                flow.point_data["Density"],
            )
        )
    ).T

    qoi_names = ["Pressure", "Velocity_x", "Velocity_y", "Pressure_Coefficient"]

    qois_mean = torch.mean(qois, dim=0)
    # qois_mean
    qois_std = torch.std(qois, dim=0)
    # qois_std

    qois_scaled = (qois - qois_mean) / qois_std

    u_bc = torch.zeros((node_type_ids.shape[0], qois.shape[1]))

    INC_DENSITY_INIT = 998.2

    u_bc[:, 4] = INC_DENSITY_INIT

    nodes_field = np.where(node_type_ids == 0)[0]

    nodes_velocity_inlet = np.where(node_type_ids == 0)[0]
    u_bc[nodes_velocity_inlet, 1] = 1.775  # x-velocity

    nodes_pressure_outlet = np.where(node_type_ids == 0)[0]
    u_bc[nodes_pressure_outlet, 0] = 0.0  # pressure

    nodes_euler_wall = np.where(node_type_ids == 0)[0]

    u_bc_scaled = (u_bc - qois_mean) / qois_std

    return (
        qoi_names,
        qois,
        u_bc,
        qois_scaled,
        u_bc_scaled,
        qois_mean,
        qois_std,
        flow.points[:, :D],
    )


def sample_points(simplices, node_pts, n_samples=10_000, replace=True):
    N, D = np.shape(simplices)

    if replace:
        i_samples = np.random.choice(N, n_samples, replace=True)
    else:
        # if n_samples > N:
        a = n_samples // N
        b = n_samples % N
        i_samples = [np.arange(N) for i in range(a)]
        if b > 0:
            i_samples.append(np.random.choice(N, b, replace=False))
        i_samples = np.concatenate(i_samples)

        # i_samples = np.random.choice(N, b, replace=False)
        # for i in range(a):
        #     i_samples = np.concatenate((i_samples, np.arange(N)))

    w_samples = np.random.rand(n_samples, D)
    w_samples = w_samples / w_samples.sum(axis=1)[:, np.newaxis]

    x_samples = np.sum(
        node_pts[simplices[i_samples]] * w_samples[:, :, np.newaxis], axis=1
    )

    return x_samples, i_samples


def find_containing_simplices(tris, x_samples, node_pts, node_lvls, dmin=5e-06):

    simplex_indices = torch.empty((len(x_samples), len(tris)), dtype=torch.long)
    simplex_node_ids = []
    for lvl, tri in enumerate(tris):
        i_min = np.min(np.where(node_lvls == lvl)[0])
        simplex_inds = tri.find_simplex(x_samples)

        for i in range(1000):
            inds_invalid = np.where(simplex_inds < 0)[0]
            N, D = x_samples[inds_invalid].shape
            if N == 0:
                break
            simplex_inds[inds_invalid] = tri.find_simplex(
                x_samples[inds_invalid] + dmin * np.random.rand(N, D)
            )
            if i == 999:
                print("WARNING: could not find simplex for", inds_invalid)

        simplex_indices[:, lvl] = torch.tensor(
            simplex_inds, dtype=torch.long
        )  # + i_min

        simplex_node_ids.append(torch.tensor(tri.vertices, dtype=torch.long) + i_min)

    return simplex_indices, simplex_node_ids


# def get_data(
#     node_pts,
#     x_nodes,
#     u_nodes,
#     edges,
#     node_type_ids,
#     node_lvls,
#     cells,
#     tri_transforms,
#     x_data,
#     x_data_type,
#     # x_data_bc,
#     x_res,
#     x_res_type,
#     # x_res_bc,
#     u_data,
#     # u_bc,
#     u_x_data,
#     # u_x_bc,
#     u_xx_data,
#     # u_xx_bc,
#     x_data_simplex_indices,
#     x_data_simplex_node_ids,
#     # x_data_simplex_transforms,
#     x_res_simplex_indices,
#     x_res_simplex_node_ids,
#     # x_res_simplex_transforms,
#     # x_bc_simplex_indices,
#     # x_bc_simplex_node_ids,
#     x_out,
#     x_out_simplex_indices,
#     x_out_simplex_node_ids,
#     node_new_ids,
# ):
#     # x_nodes == node_pts

#     data = Data(
#         # Graph
#         x=torch.tensor(x_nodes),
#         u_nodes=torch.tensor(u_nodes),
#         edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
#         # cells=None,
#         node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
#         node_lvls=torch.tensor(node_lvls, dtype=torch.long),
#         simplex_transforms=[
#             torch.tensor(tri_transform) for tri_transform in tri_transforms
#         ],
#         # Residuals
#         x_res=torch.tensor(x_res),
#         x_res_type=torch.tensor(x_res_type),
#         x_res_simplex_indices=torch.tensor(x_res_simplex_indices),
#         x_res_simplex_node_ids=x_res_simplex_node_ids,
#         # Data
#         x_data=torch.tensor(x_data),
#         x_data_type=torch.tensor(x_data_type),
#         u_data=torch.tensor(u_data),
#         # u_x_data=torch.tensor(u_x_data),
#         # u_xx_data=torch.tensor(u_xx_data),
#         x_data_simplex_indices=torch.tensor(x_data_simplex_indices),
#         x_data_simplex_node_ids=x_data_simplex_node_ids,
#         # Output
#         x_out=torch.tensor(x_out),
#         x_out_simplex_indices=torch.tensor(x_out_simplex_indices),
#         x_out_simplex_node_ids=x_out_simplex_node_ids,
#         node_new_ids=torch.tensor(node_new_ids),
#     )

#     return data


# def sample_points2(tris, node_pts, node_lvls, n_samples=10_000):
#     tri0 = tris[0]

#     # n_samples = 10_000
#     i_samples = np.random.choice(len(tri0.vertices), n_samples, replace=True)
#     w_samples = np.random.rand(n_samples, 3)
#     w_samples = w_samples / w_samples.sum(axis=1)[:, np.newaxis]

#     x_samples = np.sum(node_pts[tri0.vertices[i_samples]] * w_samples[:, :, np.newaxis], axis=1)

#     x_samples = node_pts[np.where(node_lvls==0)[0]]


#     simplex_indices = torch.empty((len(x_samples), len(tris)), dtype=torch.long)
#     simplex_node_ids = []
#     for lvl, tri in enumerate(tris):
#         i_min = np.min(np.where(node_lvls == lvl)[0])
#         simplex_indices[:, lvl] = torch.tensor(tri.find_simplex(x_samples), dtype=torch.long) #+ i_min
#         simplex_node_ids.append(torch.tensor(tri.vertices, dtype=torch.long) + i_min)

#     return x_samples, simplex_indices, simplex_node_ids


def get_data_from_simulation(
    data_dir,
    mesh_filename,
    sim_file_name="flow.vtu",
    num_levels=4,
    n_vol_sample_multiplier=1.0,
    n_bc_sample_multiplier=1.0,
    verbose=False,
):
    dim_x = 2

    if verbose:
        print("Loading mesh")

    mesh, _ = load_mesh(data_dir, mesh_filename)

    n_vol_samples0 = int(n_vol_sample_multiplier * len(mesh.cells[0]))
    if verbose:
        print(n_vol_samples0)

    x_vol0, _ = sample_points(
        mesh.cells[0].data, mesh.points, n_vol_samples0, replace=False
    )

    n_vol_samples1 = int(n_vol_sample_multiplier * len(mesh.cells[1]))
    if verbose:
        print(n_vol_samples0)

    x_vol1, _ = sample_points(
        mesh.cells[1].data, mesh.points, n_vol_samples1, replace=False
    )
    x_vol = np.concatenate((x_vol0, x_vol1), axis=0)

    n_vol_samples = n_vol_samples0 + n_vol_samples1
    node_type_ids = [np.zeros(n_vol_samples, dtype=np.int32)]
    node_su2_tags = [np.zeros(n_vol_samples, dtype=np.int32)]

    x_bc = []
    i_bc = []

    for su2_tag in np.unique(mesh.cell_data["su2:tag"][2]):
        cells = mesh.cells[2].data[mesh.cell_data["su2:tag"][2] == su2_tag]
        if verbose:
            print(su2_tag, len(cells))

        n_bc_samples = int(n_bc_sample_multiplier * len(cells))
        x_bc_i, i_bc_i = sample_points(cells, mesh.points, n_bc_samples, replace=False)
        x_bc.append(x_bc_i)
        i_bc.append(i_bc_i)

        node_type_ids.append(
            np.ones(n_bc_samples, dtype=np.int32) * su2_to_node_type[su2_tag]
        )
        node_su2_tags.append(np.ones(n_bc_samples, dtype=np.int32) * su2_tag)

    # hull = scipy.spatial.ConvexHull(mesh.points)
    # x_hull = mesh.points[hull.vertices]
    x_hull = np.array(
        [
            [-120, 120],
            [-120, 40],
            [-120, -40],
            [-120, -120],
            [-40, 120],
            [40, 120],
            [-40, -120],
            [40, -120],
            [120, 120],
            [120, 40],
            [120, -40],
            [120, -120],
        ]
    )
    node_type_ids.append(np.ones(len(x_hull), dtype=np.int32) * node_types["bbox"])
    node_su2_tags.append(np.ones(len(x_hull), dtype=np.int32) * 6)

    x_bc = np.concatenate(x_bc)
    i_bc = np.concatenate(i_bc)
    x0 = np.concatenate((x_vol, x_bc, x_hull))
    # x0 = np.concatenate((x_vol, x_bc))
    node_type_ids = np.concatenate(node_type_ids)
    node_su2_tags = np.concatenate(node_su2_tags)

    if verbose:
        print("Creating graph")

    (
        node_pts,
        node_orig_ids,
        node_new_ids,
        node_type_ids,
        node_lvls,
        node_su2_tags,
        edges,
        node_subsampling_prob,
        tris,
        cells,
    ) = create_multilevel_graph(x0, node_type_ids, node_su2_tags, num_levels=num_levels)

    if verbose:
        print("Loading simulation")

    # sim_file_name = "flow.vtu"
    qoi_names, qois, qois_scaled, qois_mean, qois_std, flow_points = load_sim_data_RANS(  # type: ignore
        data_dir, sim_file_name, node_type_ids
    )

    dim_u = len(qoi_names)

    x_res0, _ = sample_points(
        mesh.cells[0].data, mesh.points, n_samples=10_000, replace=False
    )
    x_res_type0 = [np.zeros(x_res0.shape[0], dtype=np.int8)]
    x_res1, _ = sample_points(
        mesh.cells[1].data, mesh.points, n_samples=5_000, replace=False
    )
    x_res_type1 = [np.zeros(x_res1.shape[0], dtype=np.int8)]
    x_res = np.concatenate((x_res0, x_res1), axis=0)
    # x_res_type = np.concatenate((x_res_type0, x_res_type1), axis=0)
    x_res_type = x_res_type0 + x_res_type1

    x_res_bc = []
    x_res_bc_type = []
    x_res_bc_sample_inds = []
    for su2_tag in np.unique(mesh.cell_data["su2:tag"][2]):
        cells = mesh.cells[2].data[mesh.cell_data["su2:tag"][2] == su2_tag]
        n_res_bc_samples = 10 * len(cells)
        if verbose:
            print(su2_tag, len(cells), n_res_bc_samples)
        x_res_bc_i, inds = sample_points(
            cells, mesh.points, n_res_bc_samples, replace=False
        )
        x_res_bc.append(x_res_bc_i)
        x_res_type_i = np.ones(n_res_bc_samples, dtype=np.int8) * su2_tag
        x_res_bc_type.append(x_res_type_i)
        x_res_bc_sample_inds.append(inds)

    # x_res = np.concatenate([x_res] + x_res_bc)
    x_res_type = np.concatenate(x_res_type)

    x_res_simplex_indices, x_res_simplex_node_ids = find_containing_simplices(
        tris, x_res, node_pts, node_lvls
    )

    x_res_bc = np.concatenate(x_res_bc)
    x_res_bc_type = np.concatenate(x_res_bc_type)
    x_res_bc_sample_inds = np.concatenate(x_res_bc_sample_inds)

    u_res_bc = np.zeros((x_res_bc.shape[0], qois.shape[1]))
    u_res_bc_mask = np.zeros((x_res_bc.shape[0], qois.shape[1]), dtype=float)

    airfoil_points = mesh.points[mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == 1]]

    d = airfoil_points[:, 1] - airfoil_points[:, 0]
    d_norm = np.linalg.norm(d, axis=1)
    rot = np.array([[0, 1], [-1, 0]])
    n = d @ rot / d_norm[:, np.newaxis]

    config = process_config(data_dir + "/..", "config_DSN.cfg")

    Ma = float(config["MACH_NUMBER"])
    AOA = float(config["AOA"])
    Re = float(config["REYNOLDS_NUMBER"])

    kappa = 1.4
    R_S = 287.058
    T_ref = 288.15

    v_ref = np.sqrt(kappa * R_S * T_ref) * Ma
    v_ref_x = v_ref * np.cos(AOA / 180.0 * np.pi)
    v_ref_y = v_ref * np.sin(AOA / 180.0 * np.pi)

    mu = 1.7893e-005  # viscosity
    Re = 6.5e006

    rho_ref = Re * mu / v_ref

    # airfoil
    m1 = x_res_bc_type == 1
    u_res_bc_mask[m1, 3:5] = 1
    u_res_bc[m1, 3:5] = 0.0

    # farfield
    m2 = x_res_bc_type == 2
    u_res_bc_mask[m2, 0] = 1  # Density
    u_res_bc_mask[m2, 2] = 1  # Mach
    u_res_bc_mask[m2, 3] = 1  # Momentum_x
    u_res_bc_mask[m2, 4] = 1  # Momentum_y
    u_res_bc_mask[m2, 6] = 1  # Temperature

    u_res_bc[m2, 0] = rho_ref  # Density
    u_res_bc[m2, 2] = Ma  # Mach
    u_res_bc[m2, 3] = rho_ref * v_ref_x  # Momentum_x
    u_res_bc[m2, 4] = rho_ref * v_ref_y  # Momentum_y
    u_res_bc[m2, 6] = T_ref  # Temperature

    if verbose:
        print("u_res_bc_mask.sum():", u_res_bc_mask.sum())

    x_res_bc_simplex_indices, x_res_bc_simplex_node_ids = find_containing_simplices(
        tris, x_res_bc, node_pts, node_lvls
    )
    # x_res_simplex_transforms = [tri.transform for tri in tris]

    from sklearn.model_selection import train_test_split

    u = qois
    # u = qois_scaled

    data_inds, data_inds_validation = train_test_split(
        np.arange(u.shape[0]), test_size=0.5
    )
    # data_inds = np.where()

    x_out_validation = flow_points[data_inds_validation]
    u_out_validation = u[data_inds_validation]

    val_data = (data_inds_validation, x_out_validation, u_out_validation)

    x_data = flow_points[data_inds]
    u_data = u[data_inds]
    x_data_type = np.ones(data_inds.shape[0], dtype=np.int8) * 0

    # SDF (Signed Distance Function)
    cells = mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == 1]
    # x_sdf_data, i_sdf_data = sample_points(cells, mesh.points, 1_000, replace=False)
    # u_sdf_data = np.zeros(x_sdf_data.shape[0])

    airfoil_points = mesh.points[mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == 1]]

    d = airfoil_points[:, 1] - airfoil_points[:, 0]
    d_norm = np.linalg.norm(d, axis=1)
    rot = np.array([[0, 1], [-1, 0]])
    n = d @ rot / d_norm[:, np.newaxis]

    # u_x_sdf_data = n[i_sdf_data]

    # node_su2_tags

    u_nodes = torch.zeros((node_pts.shape[0], u.shape[1]))

    # INC_DENSITY_INIT= 998.2

    # u_nodes[:,4] = INC_DENSITY_INIT

    # nodes_field = np.where(node_su2_tags==0)[0]

    # nodes_velocity_inlet = np.where(node_su2_tags == 3)[0]
    # u_nodes[nodes_velocity_inlet, 1] = 1.775  # x-velocity

    # nodes_pressure_outlet = np.where(node_su2_tags == 4)[0]
    # u_nodes[nodes_pressure_outlet, 0] = 0.0  # pressure

    # airfoil
    m1 = node_su2_tags == 1
    u_nodes[m1, 3:5] = 0.0  # Momentum

    # farfield
    m2 = node_su2_tags == 2
    u_nodes[m2, 0] = rho_ref  # Density
    u_nodes[m2, 2] = Ma  # Mach
    u_nodes[m2, 3] = rho_ref * v_ref_x  # Momentum_x
    u_nodes[m2, 4] = rho_ref * v_ref_y  # Momentum_y
    u_nodes[m2, 6] = T_ref  # Temperature

    # nodes_euler_wall = np.where(node_type_ids==0)[0]

    # u_nodes = (u_nodes - qois_mean) / qois_std

    x_data_simplex_indices, x_data_simplex_node_ids = find_containing_simplices(
        tris, x_data, node_pts, node_lvls
    )

    x_out_simplex_indices, x_out_simplex_node_ids = find_containing_simplices(
        tris, x_out_validation, node_pts, node_lvls
    )

    if verbose:
        print("Creating GNN Data object")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # print("Device:", device)
    # model = PIGNN_Euler(device=device, num_residual_latent_updates=12).to(device)

    x_nodes = node_pts
    u_x_data = None
    u_xx_data = None
    tri_transforms = [tri.transform for tri in tris]

    # one_data = get_data(
    #     node_pts,
    #     x_nodes,
    #     u_nodes,
    #     edges,
    #     node_type_ids,
    #     node_lvls,
    #     cells,
    #     tri_transforms,
    #     x_data,
    #     x_data_type,
    #     x_res,
    #     x_res_type,
    #     u_data,
    #     u_x_data,
    #     u_xx_data,
    #     x_data_simplex_indices,
    #     x_data_simplex_node_ids,
    #     x_res_simplex_indices,
    #     x_res_simplex_node_ids,
    #     x_out=x_out_validation,
    #     x_out_simplex_indices=x_out_simplex_indices,
    #     x_out_simplex_node_ids=x_out_simplex_node_ids,
    #     node_new_ids=node_new_ids,
    # )  # .to(device)

    # data = Data(
    #     # Graph
    #     x=torch.tensor(x_nodes).reshape(1, -1, dim_x),
    #     u_nodes=torch.tensor(u_nodes).reshape(1, -1, dim_u),
    #     edge_index=torch.tensor(edges, dtype=torch.long)
    #     .t()
    #     .reshape(1, 2, -1)
    #     .contiguous(),
    #     # cells=None,
    #     node_type_ids=torch.tensor(node_type_ids, dtype=torch.long).reshape(1, -1),
    #     node_lvls=torch.tensor(node_lvls, dtype=torch.long).reshape(1, -1),
    #     simplex_transforms=[
    #         [torch.tensor(tri_transform) for tri_transform in tri_transforms]
    #     ],
    #     # Residuals
    #     x_res=torch.tensor(x_res).reshape(1, -1, dim_x),
    #     x_res_type=torch.tensor(x_res_type).reshape(1, -1),
    #     x_res_simplex_indices=torch.tensor(x_res_simplex_indices).reshape(
    #         1, -1, num_levels
    #     ),
    #     x_res_simplex_node_ids=[x_res_simplex_node_ids],
    #     # Data
    #     x_data=torch.tensor(x_data).reshape(1, -1, dim_x),
    #     x_data_type=torch.tensor(x_data_type).reshape(1, -1),
    #     u_data=torch.tensor(u_data).reshape(1, -1, dim_u),
    #     # u_x_data=torch.tensor(u_x_data),
    #     # u_xx_data=torch.tensor(u_xx_data),
    #     x_data_simplex_indices=torch.tensor(x_data_simplex_indices).reshape(
    #         1, -1, num_levels
    #     ),
    #     x_data_simplex_node_ids=[x_data_simplex_node_ids],
    #     # Output
    #     x_out=torch.tensor(x_out_validation).reshape(1, -1, dim_x),
    #     x_out_simplex_indices=torch.tensor(x_out_simplex_indices).reshape(
    #         1, -1, num_levels
    #     ),
    #     x_out_simplex_node_ids=[x_out_simplex_node_ids],
    #     node_new_ids=torch.tensor(node_new_ids).reshape(1, -1),
    #     inds_validation=torch.tensor(data_inds_validation).reshape(1, -1),
    #     x_out_validation=torch.tensor(x_out_validation).reshape(1, -1, dim_x),
    #     u_out_validation=torch.tensor(u_out_validation).reshape(1, -1, dim_u),
    #     qois_mean=torch.tensor(qois_mean).reshape(1, dim_u),
    #     qois_std=torch.tensor(qois_std).reshape(1, dim_u),
    #     x_res_bc=torch.tensor(x_res_bc).reshape(1, -1, dim_x),
    #     x_res_bc_type=torch.tensor(x_res_bc_type).reshape(1, -1),
    #     u_res_bc=torch.tensor(u_res_bc).reshape(1, -1, dim_u),
    #     u_res_bc_mask=torch.tensor(u_res_bc_mask).reshape(1, -1, dim_u),
    #     x_res_bc_simplex_indices=torch.tensor(x_res_bc_simplex_indices).reshape(
    #         1, -1, num_levels
    #     ),
    #     x_res_bc_simplex_node_ids=[x_res_bc_simplex_node_ids],
    #     data_dir=data_dir,
    #     Ma=torch.tensor([Ma]),
    #     Re=torch.tensor([Re]),
    #     AOA=torch.tensor([AOA]),
    # )

    data = Data(
        # Graph
        x=torch.tensor(x_nodes),
        u_nodes=torch.tensor(u_nodes),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        # cells=None,
        node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
        node_lvls=torch.tensor(node_lvls, dtype=torch.long),
        simplex_transforms=[
            torch.tensor(tri_transform) for tri_transform in tri_transforms
        ],
        # Residuals
        x_res=torch.tensor(x_res),
        x_res_type=torch.tensor(x_res_type),
        x_res_simplex_indices=torch.tensor(x_res_simplex_indices),
        x_res_simplex_node_ids=x_res_simplex_node_ids,
        # Data
        x_data=torch.tensor(x_data),
        x_data_type=torch.tensor(x_data_type),
        u_data=torch.tensor(u_data),
        # u_x_data=torch.tensor(u_x_data),
        # u_xx_data=torch.tensor(u_xx_data),
        x_data_simplex_indices=torch.tensor(x_data_simplex_indices),
        x_data_simplex_node_ids=x_data_simplex_node_ids,
        # Output
        x_out=torch.tensor(x_out_validation),
        x_out_simplex_indices=torch.tensor(x_out_simplex_indices),
        x_out_simplex_node_ids=x_out_simplex_node_ids,
        node_new_ids=torch.tensor(node_new_ids),
        inds_validation=torch.tensor(data_inds_validation),
        x_out_validation=torch.tensor(x_out_validation),
        u_out_validation=torch.tensor(u_out_validation),
        qois_mean=torch.tensor(qois_mean),
        qois_std=torch.tensor(qois_std),
        x_res_bc=torch.tensor(x_res_bc),
        x_res_bc_type=torch.tensor(x_res_bc_type),
        u_res_bc=torch.tensor(u_res_bc),
        u_res_bc_mask=torch.tensor(u_res_bc_mask),
        x_res_bc_simplex_indices=torch.tensor(x_res_bc_simplex_indices),
        x_res_bc_simplex_node_ids=x_res_bc_simplex_node_ids,
        data_dir=data_dir,
        # Ma=torch.tensor([Ma]),
        # Re=torch.tensor([Re]),
        # AOA=torch.tensor([AOA]),
        Ma_ref=torch.ones_like(u_nodes[:,:1]) * Ma,
        Re_ref=torch.ones_like(u_nodes[:,:1]) * Ma,
        AOA_ref=torch.ones_like(u_nodes[:,:1]) * Ma,
    )

    return data


### hier weitermachen
### diese Funktionen für Erstellung eines Datasets nutzen
### dann Residuals mit autodiff berechnen
### Einteilung der x,u in data, bc und residuals
### dann auf cluster rechnen
### dann evtl. auf pytorch-lightning migrieren
