# ---------------------------------------------------------------------------
# wave prediction with mslp + wind using a transformer encoder
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------


import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# spatial CNN encorder

class SpatialCNNEncoder(nn.Module):
    """
    encodes a lat–lon field at each timestep into a compact spatial embedding.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),

            # spatial averaging
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        """
        x: (b, t, lat, lon)
        returns: (b, t, out_dim)
        """
        b, t, h, w = x.shape
        x = x.reshape(b * t, 1, h, w)  # use reshape instead of view to handle non-contiguous tensors (e.g. after slicing)
        z = self.encoder(x)
        return z.view(b, t, -1)

# model definition

class AtmosWaveTransformer(nn.Module):
    """
    architecuture: CNN (space) + Transformer (time)
    
    three-input version:
        x1: (b, t, lat, lon)  # mslp features
        x2: (b, t, lat, lon)  # u-component of wind
        x3: (b, t, lat, lon)  # v-component of wind

    output:
        (b, output_dim)  # here: 4 [hs, t0, thq_sin, thq_cos]
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.3,
        spatial_dim: int = 32,
        output_dim: int = 4,
    ):
        super().__init__()

        # shared spatial encoder
        self.spatial_encoder = SpatialCNNEncoder(out_dim=spatial_dim)

        # projection to transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(3 * spatial_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # temporal transformer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers
        )

        # prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, X_mslp, X_windU, X_windV):
        z_mslp  = self.spatial_encoder(X_mslp)
        z_windU = self.spatial_encoder(X_windU)
        z_windV = self.spatial_encoder(X_windV)

        z = torch.cat([z_mslp, z_windU, z_windV], dim=-1)
        z = self.input_proj(z)

        z = self.transformer(z)

        # mean pooling over time
        z = z.mean(dim=1)
        return self.head(z)

#hyperparameter loader
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
):
    """
    trains the transformer model and saves a checkpoint that can be reloaded later.

    expected shapes:
        X_mslp:  (n, t, lat, lon)
        X_windU: (n, t, lat, lon)
        X_windV: (n, t, lat, lon)
        y:       (n, 4)      # [hs, t0, thq_sin, thq_cos]
    """

    os.makedirs(models_dir, exist_ok=True)

    # set seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    # infer sequence length and feature dimensions from tensors
    N, T, lat, lon = X_mslp.shape
    _, T_u, _, _ = X_windU.shape
    _, T_v, _, _ = X_windV.shape

    # simple consistency checks
    assert T == T_u == T_v, "sequence length mismatch between inputs"
    assert N == X_windU.shape[0] == X_windV.shape[0] == y.shape[0], "sample count mismatch"

    print(f"inferred sequence length T = {T}")
    print(f"inferred feature count N = {N}")

    # sequential split to avoid leakage
    ratio = 0.7894620698765736  # ratio chosen so that the split occurs exactly at 2007/2008, free to use 0.8.
    n_train = int(0.8166789314257354 * ratio * N) # ratio chosen so that the split occurs exactly at 1996/1997, free to use 0.8.
    n_val = int(ratio * N)
 
    # get testing date stamps
    test_dates = y_dates[n_val:]      # list of strings for test period
    
    print(f"test period begins at: {test_dates[0]}")
    print(f"test period ends   at: {test_dates[-1]}")

    # mslp splits
    X_mslp_tr, X_mslp_va, X_mslp_te = (
        X_mslp[:n_train],
        X_mslp[n_train:n_val],
        X_mslp[n_val:],
    )

    # wind u splits
    X_windU_tr, X_windU_va, X_windU_te = (
        X_windU[:n_train],
        X_windU[n_train:n_val],
        X_windU[n_val:],
    )

    # wind v splits
    X_windV_tr, X_windV_va, X_windV_te = (
        X_windV[:n_train],
        X_windV[n_train:n_val],
        X_windV[n_val:],
    )

    # target splits
    y_tr, y_va, y_te = y[:n_train], y[n_train:n_val], y[n_val:]

    # normalise target with train statistics only
    y_mean = y_tr.mean(dim=0, keepdim=True)
    y_std = y_tr.std(dim=0, keepdim=True).clamp_min(1e-6)

    y_tr = (y_tr - y_mean) / y_std
    y_va = (y_va - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std

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

    # load hyperparameters from txt file (no defaults in code)
    hp_path = "../config/transformer_hyperparams.txt"
    hp = load_hyperparams(hp_path)

    # unpack into variables (all keys must be present in the file)
    d_model = hp["d_model"]
    nhead = hp["nhead"]
    num_layers = hp["num_layers"]
    d_ff = hp["d_ff"]
    dropout = hp["dropout"]
    spatial_dim = hp["spatial_dim"]
    l_rate = hp["l_rate"]
    weight_decay = hp["weight_decay"]
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
    print(f"  spatial_dim    = {spatial_dim}")
    print(f"  l_rate         = {l_rate}")
    print(f"  weight_decay   = {weight_decay}")
    print(f"  max_epochs     = {max_epochs}")
    print(f"  patience       = {patience}")
    print(f"  patience_limit = {patience_limit}")

    # model, loss, optimiser, scheduler
    model = AtmosWaveTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        spatial_dim=spatial_dim,
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
        max_lr=2e-2,
        steps_per_epoch=len(dl_train),
        epochs=max_epochs,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses, val_losses = [], []

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

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        print(f"restored best model from epoch {best_state['epoch'] + 1}")

    # prepare checkpoint for saving
    today = datetime.today().strftime("%Y_%m_%d")
    ckpt_path = os.path.join(models_dir, f"{today}_mslp_wind_transformer_4d.pt")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "AtmosWaveTransformer",
        "hyperparams": {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "dropout": dropout,
            "spatial_dim": spatial_dim,
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
    """
    direct execution:
    - loads pre-built tensors from disk
    - trains the transformer model
    - saves a checkpoint to ../models
    """
    # path to the prepared tensors saved earlier
    tensor_ckpt_path = Path("../data/prepared_tensors/wave_mslp_wind_tensors_4d.pt")

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
    
    # list of timestamp strings
    y_dates = ckpt["y_dates"]

    # optional: basic alignment check
    assert (
        X_mslp.shape[0]
        == X_windU.shape[0]
        == X_windV.shape[0]
        == y.shape[0]
    ), "mismatch in number of samples between inputs and target."

    print("tensor shapes:")
    print("  X_mslp :", X_mslp.shape)
    print("  X_windU:", X_windU.shape)
    print("  X_windV:", X_windV.shape)
    print("  y      :", y.shape)

    # run training
    train_and_save_model(
        X_mslp=X_mslp,
        X_windU=X_windU,
        X_windV=X_windV,
        y=y,
        y_dates=y_dates,
        models_dir="../models",
    )