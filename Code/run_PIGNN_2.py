from PIGNN_Model_v1 import *
from prepare_data import *

import scipy.spatial

print("Starting")

data_dir = "../Data/Inc_Inviscid_Hydrofoil"
mesh_filename = "mesh_NACA0012_5deg_6814.su2"


print("Loading mesh")

mesh, _ = load_mesh(data_dir, mesh_filename)

n_vol_samples = 1 * len(mesh.points)
print(n_vol_samples)

x_vol, _ = sample_points(mesh.cells[0].data, mesh.points, n_vol_samples, replace=False)
node_type_ids = [np.zeros(n_vol_samples, dtype=np.int32)]
node_su2_tags = [np.zeros(n_vol_samples, dtype=np.int32)]

x_bc = []
i_bc = []

for su2_tag in np.unique(mesh.cell_data["su2:tag"][1]):
    cells = mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == su2_tag]
    print(su2_tag, len(cells))

    n_bc_samples = 2 * len(cells)
    x_bc_i, i_bc_i = sample_points(cells, mesh.points, n_bc_samples, replace=False)
    x_bc.append(x_bc_i)
    i_bc.append(i_bc_i)

    node_type_ids.append(
        np.ones(n_bc_samples, dtype=np.int32) * su2_to_node_type[su2_tag]
    )
    node_su2_tags.append(np.ones(n_bc_samples, dtype=np.int32) * su2_tag)

hull = scipy.spatial.ConvexHull(mesh.points)
x_hull = mesh.points[hull.vertices]
node_type_ids.append(np.ones(len(x_hull), dtype=np.int32) * node_types["bbox"])
node_su2_tags.append(np.ones(len(x_hull), dtype=np.int32) * 6)

x_bc = np.concatenate(x_bc)
i_bc = np.concatenate(i_bc)
x0 = np.concatenate((x_vol, x_bc, x_hull))
node_type_ids = np.concatenate(node_type_ids)
node_su2_tags = np.concatenate(node_su2_tags)


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
) = create_multilevel_graph(x0, node_type_ids, node_su2_tags, num_levels=8)


print("Loading simulation")

sim_file_name = "flow.vtu"
qoi_names, qois, u_bc, qois_scaled, u_bc_scaled, qois_mean, qois_std, flow_points = load_sim_data_euler(  # type: ignore
    data_dir, sim_file_name, node_type_ids
)


x_res, _ = sample_points(
    mesh.cells[0].data, mesh.points, n_samples=10_000, replace=False
)
x_res_type = [np.zeros(x_res.shape[0], dtype=np.int8)]

x_res_bc = []
for su2_tag in np.unique(mesh.cell_data["su2:tag"][1]):
    cells = mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == su2_tag]
    n_res_bc_samples = 4 * len(cells)
    print(su2_tag, len(cells), n_res_bc_samples)
    x_res_bc_i, _ = sample_points(cells, mesh.points, n_res_bc_samples, replace=False)
    x_res_bc.append(x_res_bc_i)
    x_res_type_i = np.ones(n_res_bc_samples, dtype=np.int8) * su2_tag
    x_res_type.append(x_res_type_i)

x_res = np.concatenate([x_res] + x_res_bc)
x_res_type = np.concatenate(x_res_type)


x_res_simplex_indices, x_res_simplex_node_ids = find_containing_simplices(
    tris, x_res, node_pts, node_lvls
)
x_res_simplex_transforms = [tri.transform for tri in tris]


from sklearn.model_selection import train_test_split

u = qois
# u = qois_scaled

data_inds, data_inds_validation = train_test_split(np.arange(u.shape[0]), test_size=0.5)

x_out_validation = flow_points[data_inds_validation]
u_out_validation = u[data_inds_validation]

x_data = flow_points[data_inds]
u_data = u[data_inds]
x_data_type = np.ones(data_inds.shape[0], dtype=np.int8) * 0

# SDF (Signed Distance Function)
cells = mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == 1]
x_sdf_data, i_sdf_data = sample_points(cells, mesh.points, 1_000, replace=False)
u_sdf_data = np.zeros(x_sdf_data.shape[0])


airfoil_points = mesh.points[mesh.cells[1].data[mesh.cell_data["su2:tag"][1] == 1]]

d = airfoil_points[:, 1] - airfoil_points[:, 0]
d_norm = np.linalg.norm(d, axis=1)
rot = np.array([[0, 1], [-1, 0]])
n = d @ rot / d_norm[:, np.newaxis]


u_x_sdf_data = n[i_sdf_data]

# node_su2_tags

u_nodes = torch.zeros((node_pts.shape[0], u.shape[1]))

# INC_DENSITY_INIT= 998.2

# u_nodes[:,4] = INC_DENSITY_INIT

# nodes_field = np.where(node_su2_tags==0)[0]

nodes_velocity_inlet = np.where(node_su2_tags == 3)[0]
u_nodes[nodes_velocity_inlet, 1] = 1.775  # x-velocity

nodes_pressure_outlet = np.where(node_su2_tags == 4)[0]
u_nodes[nodes_pressure_outlet, 0] = 0.0  # pressure

# nodes_euler_wall = np.where(node_type_ids==0)[0]


# u_nodes = (u_nodes - qois_mean) / qois_std

x_data_simplex_indices, x_data_simplex_node_ids = find_containing_simplices(
    tris, x_data, node_pts, node_lvls
)

x_out_simplex_indices, x_out_simplex_node_ids = find_containing_simplices(
    tris, x_out_validation, node_pts, node_lvls
)


print("Creating GNN Data object")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print("Device:", device)
# model = PIGNN_Euler(device=device, num_residual_latent_updates=12).to(device)

x_nodes = node_pts
u_x_data = None
u_xx_data = None
tri_transforms = [tri.transform for tri in tris]

one_data = get_data(
    node_pts,
    x_nodes,
    u_nodes,
    edges,
    node_type_ids,
    node_lvls,
    cells,
    tri_transforms,
    x_data,
    x_data_type,
    x_res,
    x_res_type,
    u_data,
    u_x_data,
    u_xx_data,
    x_data_simplex_indices,
    x_data_simplex_node_ids,
    x_res_simplex_indices,
    x_res_simplex_node_ids,
    x_out=x_out_validation,
    x_out_simplex_indices=x_out_simplex_indices,
    x_out_simplex_node_ids=x_out_simplex_node_ids,
).to(device)

train_data = one_data.to(device)  # type: ignore


print("Starting training")

model = PIGNN_Euler(device=device, num_residual_latent_updates=12).to(device)

# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=5e-4)

loss_hist = []

u_out_validation = qois_scaled[data_inds_validation].to(device)[:, :4]
# u_out_validation = qois[data_inds_validation].to(device)


model.train()
for epoch in range(100_000):
    optimizer.zero_grad()
    # out = model(data)
    # loss = F.mse_loss(out, data.y)
    loss, loss_data, loss_res = model.compute_loss(train_data, l_res=0.001)  # type: ignore
    loss.backward()
    optimizer.step()

    u_pred = model(train_data)
    u_pred = (u_pred - qois_mean.to(device)) / qois_std.to(device)
    val_loss = F.mse_loss(u_pred.detach()[:, :4], u_out_validation).detach()

    loss_hist.append(loss.detach())

    if epoch % 100 == 0:
        loss_hist = torch.tensor(loss_hist)
        print(
            epoch,
            val_loss.cpu().numpy(),
            torch.mean(loss_hist).cpu().numpy(),
            torch.std(loss_hist).cpu().numpy(),
            loss.detach().cpu().numpy(),
            loss_data.detach().cpu().numpy(),
            loss_res.detach().cpu().numpy(),
            optimizer.param_groups[0]["lr"],
        )
        loss_hist = []


import os

s = "pignn_model_"
existing_models = [fn[12:15] for fn in os.listdir() if fn.startswith(s)]
if existing_models:
    new_model_name = s + "{:03d}".format(int(sorted(existing_models)[-1]) + 1) + ".pt"
else:
    new_model_name = s + "000" + ".pt"

print("Saving model as", new_model_name)
torch.save(model.state_dict(), new_model_name)
