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
    0: node_types["field"],             # volume field
    1: node_types["euler_wall"],        # airfoil
    2: node_types["euler_wall"],        # lower wall
    3: node_types["velocity_inlet"],    # inlet
    4: node_types["pressure_outlet"],   # outlet
    5: node_types["euler_wall"],        # upper wall
}




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



def create_node_graph(mesh, node_su2_tags):

    farfield_node_indices = np.where(np.array(node_su2_tags) >= 2)[0]
    airfoil_node_indices = np.where(np.array(node_su2_tags) == 1)[0]

    bbox_node_indices = [
        farfield_node_indices[np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[1],[1]]))],
        farfield_node_indices[np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[1],[-1]]))],
        farfield_node_indices[np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[-1],[1]]))],
        farfield_node_indices[np.argmax(mesh.points[farfield_node_indices, :2] @ np.array([[-1],[-1]]))],
    ]

    airfoil_edge_node_indices = [
        airfoil_node_indices[np.argmax(mesh.points[airfoil_node_indices, :2] @ np.array([[1],[0]]))],
        airfoil_node_indices[np.argmax(mesh.points[airfoil_node_indices, :2] @ np.array([[-1],[0]]))],
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
        node_pts += [[mesh.points[i,0], mesh.points[i,1]]]
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
    for k in range(len(indptr)-1):
        i = indptr[k]
        j = indptr[k+1]
        for m in indices[i:j]:
            edge = [nodes_lvl[k], nodes_lvl[m]]
            edges.append(edge)

    for lvl in range(1, 6):

        nodes_lvl = np.where(np.array(node_lvls) == lvl-1)[0]
        p = np.random.rand(len(nodes_lvl))
        nodes_lvl = [n for i,n in enumerate(nodes_lvl) if node_subsampling_prob[n] >= p[i]]
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
        for k in range(len(indptr)-1):
            i = indptr[k]
            j = indptr[k+1]
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

    return node_pts, node_orig_ids, node_new_ids, node_type_ids, node_lvls, edges, node_subsampling_prob, tris, cells


def load_sim_data_euler(data_dir, file_name, node_type_ids):

    flow = pyvista.read(os.path.join(data_dir, file_name))

    data_field_names = ["Pressure", "Velocity_x", "Velocity_y", "Pressure_Coefficient", "Density"]

    # qois = [torch.tensor(flow.point_data[qoi]).reshape(13937, -1) for qoi in data_field_names]
    # qois = torch.cat(qois, axis=1)

    qois = torch.tensor(np.stack((
        flow.point_data["Pressure"],
        flow.point_data["Velocity"][:,0],
        flow.point_data["Velocity"][:,1],
        flow.point_data["Pressure_Coefficient"],
        # flow.point_data["Density"],
    ))).T


    qois_mean = torch.mean(qois, dim=0)
    # qois_mean
    qois_std = torch.std(qois, dim=0)
    # qois_std

    qois_scaled = (qois - qois_mean) / qois_std


    u_bc = torch.zeros((node_type_ids.shape[0], qois.shape[1]))

    # INC_DENSITY_INIT= 998.2

    # u_bc[:,4] = INC_DENSITY_INIT

    nodes_field = np.where(node_type_ids==0)[0]

    nodes_velocity_inlet = np.where(node_type_ids==0)[0]
    u_bc[nodes_velocity_inlet, 1] = 1.775   # x-velocity

    nodes_pressure_outlet = np.where(node_type_ids==0)[0]
    u_bc[nodes_pressure_outlet, 0] = 0. # pressure

    nodes_euler_wall = np.where(node_type_ids==0)[0]

    u_bc_scaled = (u_bc - qois_mean) / qois_std

    return qois, u_bc, qois_scaled, u_bc_scaled, qois_mean, qois_std


def sample_points(simplices, node_pts, n_samples=10_000):
    N, D = np.shape(simplices)
    i_samples = np.random.choice(N, n_samples, replace=True)
    w_samples = np.random.rand(n_samples, D)
    w_samples = w_samples / w_samples.sum(axis=1)[:, np.newaxis]

    x_samples = np.sum(node_pts[simplices[i_samples]] * w_samples[:, :, np.newaxis], axis=1)

    return x_samples

def find_containing_simplices(tris, x_samples, node_pts, node_lvls):

    simplex_indices = torch.empty((len(x_samples), len(tris)), dtype=torch.long)
    simplex_node_ids = []
    for lvl, tri in enumerate(tris):
        i_min = np.min(np.where(node_lvls == lvl)[0])
        simplex_indices[:, lvl] = torch.tensor(tri.find_simplex(x_samples), dtype=torch.long) #+ i_min
        simplex_node_ids.append(torch.tensor(tri.vertices, dtype=torch.long) + i_min)
    
    return simplex_indices, simplex_node_ids


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



    ### hier weitermachen
    ### diese Funktionen für Erstellung eines Datasets nutzen
    ### dann Residuals mit autodiff berechnen
    ### Einteilung der x,u in data, bc und residuals
    ### dann auf cluster rechnen
    ### dann evtl. auf pytorch-lightning migrieren