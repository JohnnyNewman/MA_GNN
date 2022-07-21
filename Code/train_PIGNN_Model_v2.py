from typing import Iterable
from PIGNN_Model_v2 import *
from prepare_data_v2 import *
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader


def get_dataset(
    dsn_ids: Iterable[int],
    num_levels,
    device,
    n_vol_sample_multiplier=0.05,
    n_bc_sample_multiplier=2,
):
    mesh_filename = "mesh_RAE2822_turb_deform.su2"
    sim_file_name = "flow.vtu"
    data_dir_pattern = "../Data/SOBOL_DESIGNS_RANS/DESIGNS/DSN_{:04d}/DIRECT"
    dataset = [
        get_data_from_simulation(
            data_dir_pattern.format(i),
            mesh_filename,
            sim_file_name,
            num_levels,
            n_vol_sample_multiplier,
            n_bc_sample_multiplier,
        ).to(device)
        for i in dsn_ids
    ]

    qois_mean = torch.stack([d.qois_mean for d in dataset]).mean(dim=0)
    qois_std = torch.stack([d.qois_std for d in dataset]).mean(dim=0)
    print("qois_mean:", qois_mean)
    print("qois_std:", qois_std)

    for i in range(len(dataset)):
        dataset[i].qois_mean = qois_mean
        dataset[i].qois_std = qois_std

    return dataset


def save_model(model, dataset):
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
    torch.save(dataset, "dataset_" + new_model_name)


def train_model(
    model,
    dataset,
    device=None,
    no_levels=8,
    no_updates=12,
    max_epoch=10_001,
    start_epoch=0,
):

    loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-04)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-04, last_epoch=start_epoch - 1, verbose=False
    )

    columns = [
        "epoch",
        "data_dir",
        "loss",
        "l_data",
        "loss_data",
        "l_res",
        "loss_res",
        "l_bc",
        "loss_bc",
        "val_loss",
    ]
    results_df = pd.DataFrame(columns=columns)

    loss_hist = []
    res_history = []

    print("Starting Training")

    model.train()
    for epoch in range(start_epoch, max_epoch):
        for i, train_data in enumerate(loader):
            optimizer.zero_grad()
            model.train()
            # out = model(data)
            # loss = F.mse_loss(out, data.y)
            l_res = 0.00
            l_bc = 0.1
            l_data = 1
            loss, loss_data, loss_res, loss_bc = model.compute_loss(train_data, l_data, l_res=l_res, l_bc=l_bc, h_noise=0.1)  # type: ignore
            loss.backward()
            optimizer.step()

            model.eval()
            # u_out_validation = qois_scaled[train_data.inds_validation].to(device)[:,:]

            u_pred_scaled = model(train_data, normalized=True)
            u_out_validation_scaled = (
                train_data.u_out_validation - train_data.qois_mean
            ) / train_data.qois_std
            val_loss = F.mse_loss(
                u_pred_scaled.detach(), u_out_validation_scaled
            ).detach()

            # u_pred_scaled = (u_pred - qois_mean.to(device)) / qois_std.to(device)
            # val_loss = F.mse_loss(u_pred_scaled.detach()[:,:], u_out_validation).detach()
            # val_loss = 0

            loss_hist.append(loss.detach())

            row_df = pd.DataFrame(
                columns=columns,
                data=[
                    [
                        int(epoch),
                        str(train_data.data_dir),
                        float(loss.detach().cpu().numpy()),
                        float(l_data),
                        float(loss_data.detach().cpu().numpy()),
                        float(l_res),
                        float(loss_res.detach().cpu().numpy()),
                        float(l_bc),
                        float(loss_bc.detach().cpu().numpy()),
                        float(
                            val_loss.cpu().numpy(),
                        ),
                    ]
                ],
            )
            # store.append('results_df', row_df)

            results_df = pd.concat((results_df, row_df))

            # if epoch % 5 == 0 and i == 0:
            #     loss_hist = torch.tensor(loss_hist)
            #     l_res = 1.0
            #     l_bc = 1.0
            #     l_data = 1
            #     # print(results_df.groupby("epoch").mean().tail(1))

            #     # print(epoch,
            #     #         "val: {:.6f}".format(val_loss.cpu().numpy()),
            #     #         "mean: {:.6f}".format(torch.mean(loss_hist).cpu().numpy()),
            #     #         "std: {:.6f}".format(torch.std(loss_hist).cpu().numpy()),
            #     #         "curr: {:.6f}".format(loss.detach().cpu().numpy()),
            #     #         "data: {:.4f}".format(l_data*loss_data.detach().cpu().numpy()),
            #     #         "res: {:.4f}".format(l_res*loss_res.detach().cpu().numpy()),
            #     #         "bc: {:.4f}".format(l_bc*loss_bc.detach().cpu().numpy()),
            #     #         # "r1_l1: {:.4f}".format(model._r1_l1.detach().cpu().numpy()),
            #     #         # "r2_l1: {:.4f}".format(model._r2_l1.detach().cpu().numpy()),
            #     #         # "r3_l1: {:.4f}".format(model._r3_l1.detach().cpu().numpy()),
            #     #         # "r1_l2: {:.4f}".format(model._r1_l2.detach().cpu().numpy()),
            #     #         # "r2_l2: {:.4f}".format(model._r2_l2.detach().cpu().numpy()),
            #     #         # "r3_l2: {:.4f}".format(model._r3_l2.detach().cpu().numpy()),
            #     # )
            #     loss_hist = []
        if epoch % 1 == 0:
            # print(epoch, " ".join([f"{k}: {v.values[0]:.6f}" for k,v in results_df.groupby("epoch").mean().tail(1).items()]))
            print(
                epoch,
                " ".join(
                    [
                        f"{k}: {v:.6f}"
                        for k, v in results_df[results_df["epoch"] == epoch]
                        .mean()
                        .items()
                    ][1:]
                ),
                "lr:",
                scheduler.get_last_lr(),
            )
        #     results_df.to_excel(f"training_log/pignn_v2_training_log_epoch{epoch}.xlsx")
        #     results_df.to_csv(f"training_log/pignn_v2_training_log_epoch{epoch}.csv")
        results_df[results_df["epoch"] == epoch].to_excel(
            f"training_log/pignn_v2_training_log_epoch{epoch}.xlsx"
        )
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_checkpoint_{epoch}.pt")


if __name__ == "__main__":
    NO_LEVELS = 8
    NO_UPDATES = 12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print("Device:", device)

    dsn_ids = [(i + 12) for i in range(4)]
    dataset = get_dataset(dsn_ids, NO_LEVELS, device, 0.05, 2)

    model = PIGNN_RANS(
        device=device,
        num_residual_latent_updates=NO_UPDATES,
        num_lvls=NO_LEVELS,
        num_nodes=0,
    ).to(device)

    train_model(model, dataset, device, NO_LEVELS, NO_UPDATES, 10_001)
