#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# wave prediction with mslp + wind using a transformer encoder
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March 2026
# ---------------------------------------------------------------------------


import argparse
import os
from datetime import datetime
import time

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# model definition


class AtmosWaveTransformer(nn.Module):
    """
    three-input version:
        x1: (b, t, f1)  # mslp features
        x2: (b, t, f2)  # u-component of wind
        x3: (b, t, f3)  # v-component of wind

    output:
        (b, output_dim)  # here: 4 [hs, t0, thq_sin, thq_cos]
    """

    def __init__(
        self,
        T: int,
        F1: int,
        F2: int,
        F3: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.3,
        output_dim: int = 4,
    ):
        super().__init__()

        # encoder for first input: mslp anomaly data
        self.feature_encoder1 = nn.Sequential(
            nn.Linear(F1, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # encoder for second input: wind u10 component
        self.feature_encoder2 = nn.Sequential(
            nn.Linear(F2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # encoder for third input: wind v10 component
        self.feature_encoder3 = nn.Sequential(
            nn.Linear(F3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # positional embeddings for concatenated features
        self.pos = nn.Parameter(torch.randn(1, T, d_model * 3) * 0.02)

        # transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 3,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x1, x2, x3):
        # encode each input
        z1 = self.feature_encoder1(x1)
        z2 = self.feature_encoder2(x2)
        z3 = self.feature_encoder3(x3)

        # concatenate along feature dimension
        z = torch.cat((z1, z2, z3), dim=2)

        # add positional embeddings
        z = z + self.pos

        # transformer encoder
        z = self.transformer(z)

        # mean pooling over time
        z = z.mean(dim=1)

        # prediction head
        return self.head(z)

def load_hyperparams(path):
    hp = {}
    if not os.path.isfile(path):
        return hp
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            # try int → try float → fallback to string
            try:
                if value.lower() in {"true", "false"}:
                    hp[key] = value.lower() == "true"
                elif "." in value or "e" in value.lower():
                    hp[key] = float(value)
                else:
                    hp[key] = int(value)
            except:
                hp[key] = value
    return hp
    
def train_and_save_model(
    X_mslp: torch.Tensor,
    X_windU: torch.Tensor,
    X_windV: torch.Tensor,
    y: torch.Tensor,
    y_dates: torch.Tensor,
    models_dir: str = "../models",
    poi: str = "SEE",
    window: int = 6,
):
    """
    trains the transformer model and saves a checkpoint that can be reloaded later.

    expected shapes:
        X_mslp:  (N, T, F1)
        X_windU: (N, T, F2)
        X_windV: (N, T, F3)
        y:       (N, 4)      # [hs, t0, thq_sin, thq_cos]
    """

    os.makedirs(models_dir, exist_ok=True)

    # get date - when saving files
    today = datetime.today().strftime("%Y_%m_%d")

    # set seeds for reproducibility
    seed = args.seed   # for uncertainty [40, 41, 42, 43, 44] -- to be back to address this...
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  # these two could slow the training -- lets quantify that later! 
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # infer sequence length and feature dimensions from tensors
    N, T, F1 = X_mslp.shape
    _, T_u, F2 = X_windU.shape
    _, T_v, F3 = X_windV.shape

    # simple consistency checks
    assert T == T_u == T_v, "sequence length mismatch between inputs"
    assert N == X_windU.shape[0] == X_windV.shape[0] == y.shape[0], "sample count mismatch"

    if T != window:
        raise ValueError(
            f"requested a {window}-hour run, but the loaded tensor has T={T}"
        )

    print(f"inferred sequence length T = {T}")
    print(f"inferred feature dims: F1 (mslp) = {F1}, F2 (u) = {F2}, F3 (v) = {F3}")

    # chronological train-validation split; test data are stored separately
    y_dates = pd.DatetimeIndex(pd.to_datetime(y_dates))
    train_mask = y_dates < pd.Timestamp("1997-01-01 00:00:00")
    validation_mask = (y_dates >= pd.Timestamp("1997-01-01 00:00:00")) & (y_dates < pd.Timestamp("2008-01-01 00:00:00"))
    if not y_dates.is_monotonic_increasing:
        raise ValueError("y_dates must be sorted")
    if y_dates.max() >= pd.Timestamp("2008-01-01 00:00:00"):
        raise ValueError("trainval tensor contains test data")
    n_train, n_validation = int(train_mask.sum()), int(validation_mask.sum())
    if n_train == 0 or n_validation == 0 or n_train + n_validation != len(y_dates):
        raise ValueError("invalid train-validation split")
    print("date-based data split:", flush=True)
    print(f"  train: {n_train:,} samples, {y_dates[train_mask][0]} to {y_dates[train_mask][-1]}", flush=True)
    print(f"  validation: {n_validation:,} samples, {y_dates[validation_mask][0]} to {y_dates[validation_mask][-1]}", flush=True)
    print("  test: excluded from the training tensor", flush=True)
    X_mslp_tr, X_mslp_va = X_mslp[:n_train], X_mslp[n_train:]
    X_windU_tr, X_windU_va = X_windU[:n_train], X_windU[n_train:]
    X_windV_tr, X_windV_va = X_windV[:n_train], X_windV[n_train:]
    y_tr, y_va = y[:n_train], y[n_train:]
    y_mean = y_tr.mean(dim=0, keepdim=True)
    y_std = y_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_tr = (y_tr - y_mean) / y_std
    y_va = (y_va - y_mean) / y_std

    # dataloaders
    dl_train = DataLoader(
        TensorDataset(X_mslp_tr, X_windU_tr, X_windV_tr, y_tr),
        batch_size=1024,#1024
        shuffle=True,
    )
    dl_valid = DataLoader(
        TensorDataset(X_mslp_va, X_windU_va, X_windV_va, y_va),
        batch_size=2048,#2048
        shuffle=False,
    )

    # load hyperparameters from txt file (no defaults in code)
    hp_path = "../config/transformer_hyperparams.txt"
    hp = load_hyperparams(hp_path)

    # unpack into variables (all keys must be present in the file)
    l_rate = hp["l_rate"]
    weight_decay = hp["weight_decay"]
    dropout = hp["dropout"]
    num_layers = hp["num_layers"]
    d_model = hp["d_model"]
    d_ff = hp["d_ff"]
    nhead = hp["nhead"]
    max_epochs = hp["max_epochs"]
    patience = hp["patience"]
    patience_limit = hp["patience_limit"]

    # print hyperparameters loaded from transformer_hyperparams.txt
    print("\nhyperparameters loaded from transformer_hyperparams.txt:")
    print(f"  d_model        = {d_model}")
    print(f"  nhead          = {nhead}")
    print(f"  num_layers     = {num_layers}")
    print(f"  d_ff           = {d_ff}")
    print(f"  dropout        = {dropout}")
    print(f"  l_rate         = {l_rate}")
    print(f"  weight_decay   = {weight_decay}")
    print(f"  max_epochs     = {max_epochs}")
    print(f"  patience       = {patience}")
    print(f"  patience_limit = {patience_limit}")

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
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_parameters:,}", flush=True)
    print(f"parameters (million): {n_parameters / 1e6:.3f}", flush=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=l_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=2e-2,
        steps_per_epoch=len(dl_train),
        epochs=max_epochs,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    start_time = time.time()
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"epoch {epoch + 1:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # early stopping logic
        if val_loss < best_val_loss - patience_limit:
            best_val_loss = val_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("early stopping triggered.")
                break

    # time to train the atmoswavetransformer  
    end_time = time.time()
    total_time = end_time - start_time  
    
    # save to text file inside models directory
    file_path_tr_time = os.path.join(models_dir, f"{today}_training_time_{poi}_w{window:02d}_sd{seed:04d}.txt")
    
    with open(file_path_tr_time, "w") as f:
        f.write(f"date: {today}\n")
        f.write(f"total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
    
    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(f"restored best model from epoch {best_state['epoch'] + 1}")

    # prepare checkpoint for saving    
    ckpt_path = os.path.join(models_dir, f"{today}_atmoswave_transformer_{poi}_w{window:02d}_sd{seed:04d}.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "poi": poi,
        "window_hours": window,
        "seed":seed,
        "experiment_role": (
            "history-window sensitivity" if poi == "SEE"
            else "transferability check"
        ),
        "model_class": "AtmosWaveTransformer",
        "hyperparams": {
            "T": T,
            "F1": F1,
            "F2": F2,
            "F3": F3,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "output_dim": y.shape[1],
        },
        "normalisation": {
            "y_mean": y_mean.cpu(),
            "y_std": y_std.cpu(),
        },
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    """Load one prepared tensor file and train one transformer model."""
    parser = argparse.ArgumentParser(
        description=(
            "train the fixed transformer architecture for a selected "
            "history window; SEE is the main sensitivity experiment and "
            "ELB is reserved for the transferability check"
        )
    )
    parser.add_argument(
        "--poi",
        choices=("SEE", "ELB"),
        default="SEE",
        help=(
            "point of interest; use SEE for the 6/12/24-hour comparison "
            "and ELB only for transferability"
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=(6, 12, 24),
        required=True,
        help="atmospheric history-window length in hours",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducible training",
    )

    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("../data/prepared_tensors"),
        help="directory containing the prepared tensor files",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("../models"),
        help="directory used for model checkpoints and training-time logs",
    )
    args = parser.parse_args()

    tensor_ckpt_path = args.tensor_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_"
        f"w{args.window:02d}_trainval_1948_2007.pt"
    )

    if not tensor_ckpt_path.is_file():
        raise FileNotFoundError(
            f"could not find prepared tensor file: {tensor_ckpt_path}\n"
            "run the tensor-preparation script before model training"
        )

    print("=" * 72, flush=True)
    print("transformer training", flush=True)
    print("=" * 72, flush=True)
    print(f"poi: {args.poi}", flush=True)
    print(f"window: {args.window} h", flush=True)
    print(f"experiment: {'history-window sensitivity' if args.poi == 'SEE' else 'transferability check'}", flush=True)
    print(f"loading tensors from: {tensor_ckpt_path}", flush=True)

    # these files are created locally by the associated tensor workflow
    ckpt = torch.load(
        tensor_ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    X_mslp = ckpt["X_mslp"]
    X_windU = ckpt["X_windU"]
    X_windV = ckpt["X_windV"]
    y = ckpt["y"]
    y_dates = ckpt["y_dates"]

    assert (
        X_mslp.shape[0]
        == X_windU.shape[0]
        == X_windV.shape[0]
        == y.shape[0]
    ), "mismatch in number of samples between inputs and target"

    assert (
        X_mslp.shape[1]
        == X_windU.shape[1]
        == X_windV.shape[1]
        == args.window
    ), "tensor sequence length does not match --window"

    print("tensor shapes:", flush=True)
    print("  X_mslp :", X_mslp.shape, flush=True)
    print("  X_windU:", X_windU.shape, flush=True)
    print("  X_windV:", X_windV.shape, flush=True)
    print("  y      :", y.shape, flush=True)

    # all architecture and optimisation hyperparameters remain unchanged
    train_and_save_model(
        X_mslp=X_mslp,
        X_windU=X_windU,
        X_windV=X_windV,
        y=y,
        y_dates=y_dates,
        models_dir=str(args.models_dir),
        poi=args.poi,
        window=args.window,
    )
