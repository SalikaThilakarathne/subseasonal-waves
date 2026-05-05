# ---------------------------------------------------------------------------
# grid search for the transformer encoder deep learning model
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------


import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from train_wave_transformer import AtmosWaveTransformer


def load_tensors(tensor_ckpt_path: Path):
    """
    load pre-built tensors (x_mslp, x_windu, x_windv, y, y_dates) from disk.
    """
    if not tensor_ckpt_path.is_file():
        raise FileNotFoundError(
            f"could not find prepared tensor file: {tensor_ckpt_path}\n"
            "make sure you have run the preprocessing script that saves X_mslp, X_windU, X_windV, y."
        )

    print(f"loading tensors from: {tensor_ckpt_path}")
    ckpt = torch.load(tensor_ckpt_path, map_location="cpu")

    X_mslp = ckpt["X_mslp"]
    X_windU = ckpt["X_windU"]
    X_windV = ckpt["X_windV"]
    y = ckpt["y"]
    y_dates = ckpt["y_dates"]  # list of timestamp strings

    assert (
        X_mslp.shape[0]
        == X_windU.shape[0]
        == X_windV.shape[0]
        == y.shape[0]
        == len(y_dates)
    ), "mismatch in number of samples between inputs, target, and dates."

    print("tensor shapes:")
    print("  X_mslp :", X_mslp.shape)
    print("  X_windU:", X_windU.shape)
    print("  X_windV:", X_windV.shape)
    print("  y      :", y.shape)

    return X_mslp, X_windU, X_windV, y, y_dates


def load_grid(grid_file: Path):
    """
    load hyperparameter grid from a text file (csv with header).
    each row should define one configuration.
    """
    if not grid_file.is_file():
        raise FileNotFoundError(f"could not find grid file: {grid_file}")

    configs = []
    with grid_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # convert values to correct types
            cfg = {
                "d_model": int(row["d_model"]),
                "nhead": int(row["nhead"]),
                "num_layers": int(row["num_layers"]),
                "d_ff": int(row["d_ff"]),
                "dropout": float(row["dropout"]),
                "l_rate": float(row["l_rate"]),
                "weight_decay": float(row["weight_decay"]),
                "max_epochs": int(row["max_epochs"]),
                "patience": int(row["patience"]),
            }
            configs.append(cfg)

    if not configs:
        raise ValueError(f"no configurations found in grid file: {grid_file}")

    print(f"loaded {len(configs)} hyperparameter configurations from {grid_file}")
    return configs


def run_single_config(
    X_mslp: torch.Tensor,
    X_windU: torch.Tensor,
    X_windV: torch.Tensor,
    y: torch.Tensor,
    y_dates,
    config: dict,
    device: str = None,
):
    """
    train one atmos-wave transformer for a given hyperparameter configuration.
    returns best validation loss and epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # infer sequence length and feature dimensions from tensors
    N, T, F1 = X_mslp.shape
    _, T_u, F2 = X_windU.shape
    _, T_v, F3 = X_windV.shape

    assert T == T_u == T_v, "sequence length mismatch between inputs"
    assert N == X_windU.shape[0] == X_windV.shape[0] == y.shape[0], "sample count mismatch"

    # sequential split (your custom ratios for year-based splits)
    ratio = 0.7894620698765736     # ensures split around 2007/2008
    n_train = int(0.8166789314257354 * ratio * N)  # ensures split around 1996/1997
    n_val = int(ratio * N)

    test_dates = y_dates[n_val:]
    print(f"test period begins at: {test_dates[0]}")
    print(f"test period ends   at: {test_dates[-1]}")

    # splits
    X_mslp_tr, X_mslp_va = X_mslp[:n_train], X_mslp[n_train:n_val]
    X_windU_tr, X_windU_va = X_windU[:n_train], X_windU[n_train:n_val]
    X_windV_tr, X_windV_va = X_windV[:n_train], X_windV[n_train:n_val]
    y_tr, y_va = y[:n_train], y[n_train:n_val]

    # normalise target with train statistics only
    y_mean = y_tr.mean(dim=0, keepdim=True)
    y_std = y_tr.std(dim=0, keepdim=True).clamp_min(1e-6)

    y_tr = (y_tr - y_mean) / y_std
    y_va = (y_va - y_mean) / y_std

    # dataloaders
    dl_train = DataLoader(
        TensorDataset(X_mslp_tr, X_windU_tr, X_windV_tr, y_tr),
        batch_size=1024,
        shuffle=True,
    )
    dl_valid = DataLoader(
        TensorDataset(X_mslp_va, X_windU_va, X_windV_va, y_va),
        batch_size=2048,
        shuffle=False,
    )

    # hyper-parameters from config
    d_model = config["d_model"]
    nhead = config["nhead"]
    num_layers = config["num_layers"]
    d_ff = config["d_ff"]
    dropout = config["dropout"]
    l_rate = config["l_rate"]
    weight_decay = config["weight_decay"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]
    patience_limit = 5e-3

    # model, loss, optimiser, scheduler
    model = AtmosWaveTransformer(
        T=T,
        F1=F1,
        F2=F2,
        F3=F3,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        output_dim=y.shape[1],
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=l_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=2e-2,              # kept fixed as in your original setup
        steps_per_epoch=len(dl_train),
        epochs=max_epochs,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # training
        model.train()
        total_loss = 0.0
        total_count = 0

        for x1b, x2b, x3b, yb in dl_train:
            x1b = x1b.to(device)
            x2b = x2b.to(device)
            x3b = x3b.to(device)
            yb = yb.to(device)

            optimiser.zero_grad(set_to_none=True)
            yhat = model(x1b, x2b, x3b)
            loss = criterion(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            batch_size = x1b.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

        train_loss = total_loss / total_count

        # validation
        model.eval()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for x1b, x2b, x3b, yb in dl_valid:
                x1b = x1b.to(device)
                x2b = x2b.to(device)
                x3b = x3b.to(device)
                yb = yb.to(device)

                yhat = model(x1b, x2b, x3b)
                loss = criterion(yhat, yb)

                batch_size = x1b.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size

        val_loss = total_loss / total_count

        print(
            f"epoch {epoch + 1:03d} "
            f"| train {train_loss:.4f} "
            f"| val {val_loss:.4f}"
        )

        # early stopping
        if val_loss < best_val_loss - patience_limit:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("early stopping triggered.")
                break

    return best_val_loss, best_epoch


def main():
    """
    run grid search over hyperparameters and save results to a text file.
    """
    base_dir = Path(__file__).resolve().parent

    tensor_ckpt_path = base_dir / "../data/prepared_tensors/wave_mslp_wind_tensors.pt"
    grid_file = base_dir / "../config/transformer_grid.txt"
    results_file = base_dir / "../results/transformer_grid_search_results.txt"

    # ensure output directory exists
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # load data and grid
    X_mslp, X_windU, X_windV, y, y_dates = load_tensors(tensor_ckpt_path)
    configs = load_grid(grid_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # open results file
    with results_file.open("w") as f:
        # write header
        f.write(
            "d_model,nhead,num_layers,d_ff,dropout,l_rate,weight_decay,"
            "max_epochs,patience,best_val_loss,best_epoch\n"
        )

        # loop over configurations
        for i, cfg in enumerate(configs, start=1):
            print("\n" + "-" * 60)
            print(f"configuration {i}/{len(configs)}:")
            print(cfg)

            best_val_loss, best_epoch = run_single_config(
                X_mslp=X_mslp,
                X_windU=X_windU,
                X_windV=X_windV,
                y=y,
                y_dates=y_dates,
                config=cfg,
                device=device,
            )

            # write one line to results file
            f.write(
                f"{cfg['d_model']},{cfg['nhead']},{cfg['num_layers']},"
                f"{cfg['d_ff']},{cfg['dropout']},{cfg['l_rate']},"
                f"{cfg['weight_decay']},{cfg['max_epochs']},{cfg['patience']},"
                f"{best_val_loss:.6f},{best_epoch}\n"
            )
            f.flush()

            print(
                f"finished config {i}/{len(configs)} "
                f"| best val loss = {best_val_loss:.6f} at epoch {best_epoch}"
            )

    print(f"\nall done. results saved to: {results_file}")


if __name__ == "__main__":
    main()