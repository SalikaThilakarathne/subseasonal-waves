#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# wave prediction with spatial cnn encoders and a temporal transformer
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# October 2026
# ---------------------------------------------------------------------------

"""Train a CNN-Transformer using the existing flattened atmosphere tensors.

The tensors remain stored as (sample, time, feature). each batch is reshaped
back to its two-dimensional grid inside the model before spatial CNN encoding.

Default grids:
    mslp:  40 x 60 = 2400 cells
    wind:  20 x 20 = 400 cells

Example:
    python -u train_atmoswave_cnn_transformer.py \
        --poi SEE --window 24 --seed 42
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_END = pd.Timestamp("1997-01-01 00:00:00")
VALIDATION_END = pd.Timestamp("2008-01-01 00:00:00")


class SpatialCNNEncoder(nn.Module):
    """Encode one lat-lon field at every timestep."""

    def __init__(
        self,
        output_dim: int,
        base_channels: int = 16,
        pooled_size: int = 2,
    ) -> None:
        super().__init__()
        hidden_channels = base_channels * 2

        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((pooled_size, pooled_size)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * pooled_size * pooled_size, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Accept (batch, time, height, width), return (batch, time, dim)."""
        batch, time_steps, height, width = field.shape
        field = field.reshape(batch * time_steps, 1, height, width)
        encoded = self.projection(self.features(field))
        return encoded.reshape(batch, time_steps, -1)


class AtmosWaveCNNTransformer(nn.Module):
    """Use CNNs for space and self-attention for atmospheric history."""

    def __init__(
        self,
        sequence_length: int,
        mslp_shape: tuple[int, int],
        wind_shape: tuple[int, int],
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        base_channels: int = 16,
        pooled_size: int = 2,
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        self.mslp_shape = mslp_shape
        self.wind_shape = wind_shape

        # separate encoders avoid forcing different physical variables to share weights
        self.mslp_encoder = SpatialCNNEncoder(
            d_model, base_channels, pooled_size
        )
        self.wind_u_encoder = SpatialCNNEncoder(
            d_model, base_channels, pooled_size
        )
        self.wind_v_encoder = SpatialCNNEncoder(
            d_model, base_channels, pooled_size
        )

        temporal_dim = 3 * d_model
        self.position = nn.Parameter(
            torch.randn(1, sequence_length, temporal_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        x_mslp: torch.Tensor,
        x_wind_u: torch.Tensor,
        x_wind_v: torch.Tensor,
    ) -> torch.Tensor:
        batch, time_steps, _ = x_mslp.shape
        mslp = x_mslp.reshape(batch, time_steps, *self.mslp_shape)
        wind_u = x_wind_u.reshape(batch, time_steps, *self.wind_shape)
        wind_v = x_wind_v.reshape(batch, time_steps, *self.wind_shape)

        encoded = torch.cat(
            (
                self.mslp_encoder(mslp),
                self.wind_u_encoder(wind_u),
                self.wind_v_encoder(wind_v),
            ),
            dim=-1,
        )
        encoded = self.transformer(encoded + self.position)
        return self.head(encoded.mean(dim=1))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poi", choices=("SEE", "ELB"), default="SEE")
    parser.add_argument("--window", type=int, choices=(6, 12, 24), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor_dir", type=Path, default=Path("../data/prepared_tensors"))
    parser.add_argument("--models_dir", type=Path, default=Path("../models"))
    parser.add_argument("--config", type=Path, default=Path("../config/transformer_hyperparams.txt"))
    parser.add_argument("--mslp_height", type=int, default=40)
    parser.add_argument("--mslp_width", type=int, default=60)
    parser.add_argument("--wind_height", type=int, default=20)
    parser.add_argument("--wind_width", type=int, default=20)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--pooled_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)  #was 256
    return parser.parse_args()


def load_hyperparameters(path: Path) -> dict:
    hp = {
        "d_model": 64,
        "nhead": 8,
        "num_layers": 2,
        "d_ff": 256,
        "dropout": 0.1,
        "l_rate": 0.001,
        "weight_decay": 0.001,
        "max_epochs": 500,
        "patience": 50,
        "patience_limit": 0.0005,
    }

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = (item.strip() for item in line.split("=", 1))
        if key not in hp:
            continue
        hp[key] = type(hp[key])(value)

    return hp


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.inference_mode():
        for x1, x2, x3, target in loader:
            target = target.to(device, non_blocking=True)
            prediction = model(
                x1.to(device, non_blocking=True),
                x2.to(device, non_blocking=True),
                x3.to(device, non_blocking=True),
            )
            count = target.shape[0]
            total_loss += criterion(prediction, target).item() * count
            total_count += count

    return total_loss / total_count


def main() -> None:
    args = parse_arguments()
    set_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.models_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = args.tensor_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_"
        f"w{args.window:02d}_trainval_1948_2007.pt"
    )
    print("=" * 72, flush=True)
    print("cnn-transformer training", flush=True)
    print("=" * 72, flush=True)
    
    print(f"poi: {args.poi}", flush=True)
    print(f"window: {args.window} h", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(
        "experiment: "
        f"{'history-window sensitivity' if args.poi == 'SEE' else 'transferability check'}",
        flush=True,
    )
    print(f"loading tensors from: {tensor_path}", flush=True)

    checkpoint = torch.load(tensor_path, map_location="cpu", weights_only=False)
    x_mslp = checkpoint["X_mslp"]
    x_wind_u = checkpoint["X_windU"]
    x_wind_v = checkpoint["X_windV"]
    target = checkpoint["y"]
    dates = pd.DatetimeIndex(pd.to_datetime(checkpoint["y_dates"]))

    print("tensor shapes:", flush=True)
    print(f"  X_mslp : {x_mslp.shape}", flush=True)
    print(f"  X_windU: {x_wind_u.shape}", flush=True)
    print(f"  X_windV: {x_wind_v.shape}", flush=True)
    print(f"  y      : {target.shape}", flush=True)


    n_samples, sequence_length, mslp_features = x_mslp.shape
    wind_features = x_wind_u.shape[2]
    mslp_shape = (args.mslp_height, args.mslp_width)
    wind_shape = (args.wind_height, args.wind_width)

    print(f"using device: {device}", flush=True)
    print(
        f"inferred sequence length T = {sequence_length}",
        flush=True,
    )
    print(
        f"inferred feature dimensions: "
        f"MSLP = {mslp_features}, "
        f"wind-U = {x_wind_u.shape[2]}, "
        f"wind-V = {x_wind_v.shape[2]}",
        flush=True,
    )
    print(
        f"MSLP spatial grid: "
        f"{mslp_shape[0]} × {mslp_shape[1]}",
        flush=True,
    )
    print(
        f"wind spatial grid: "
        f"{wind_shape[0]} × {wind_shape[1]}",
        flush=True,
    )

    if np.prod(mslp_shape) != mslp_features:
        raise ValueError(
            f"mslp grid {mslp_shape} has {np.prod(mslp_shape)} cells, "
            f"but the tensor has {mslp_features} features"
        )
    if np.prod(wind_shape) != wind_features:
        raise ValueError(
            f"wind grid {wind_shape} has {np.prod(wind_shape)} cells, "
            f"but the tensor has {wind_features} features"
        )
    if sequence_length != args.window:
        raise ValueError("tensor sequence length does not match --window")

    train_indices = np.flatnonzero(dates < TRAIN_END)
    validation_indices = np.flatnonzero(
        (dates >= TRAIN_END) & (dates < VALIDATION_END)
    )
    if not dates.is_monotonic_increasing:
        raise ValueError("y_dates must be sorted")
    if dates.max() >= VALIDATION_END:
        raise ValueError("trainval tensor contains test data")
    if len(train_indices) == 0 or len(validation_indices) == 0 or len(train_indices) + len(validation_indices) != len(dates):
        raise ValueError("invalid train-validation split")
    print("date-based data split:", flush=True)
    print(f"  train: {len(train_indices):,} samples, {dates[train_indices[0]]} to {dates[train_indices[-1]]}", flush=True)
    print(f"  validation: {len(validation_indices):,} samples, {dates[validation_indices[0]]} to {dates[validation_indices[-1]]}", flush=True)
    print("  test: excluded from the training tensor", flush=True)
    train_index = torch.from_numpy(train_indices).long()

    y_train = target.index_select(0, train_index)
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0).clamp_min(1e-6)
    target_normalised = (target - y_mean) / y_std

    # views are sliced chronologically; no test data are used during training
    train_dataset = TensorDataset(
        x_mslp[:len(train_indices)],
        x_wind_u[:len(train_indices)],
        x_wind_v[:len(train_indices)],
        target_normalised[:len(train_indices)],
    )
    validation_dataset = TensorDataset(
        x_mslp[len(train_indices):],
        x_wind_u[len(train_indices):],
        x_wind_v[len(train_indices):],
        target_normalised[len(train_indices):],
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=512,  #512
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    hp = load_hyperparameters(args.config)
    print(f"hyperparameters loaded from {args.config.name}:", flush=True)
    for key, value in hp.items():
        print(f"  {key:16s} = {value}", flush=True)
    print(f"  base_channels    = {args.base_channels}", flush=True)
    print(f"  pooled_size      = {args.pooled_size}", flush=True)
    print(f"  train_batch      = {args.batch_size}", flush=True)
    print("  validation_batch = 512", flush=True)
    model = AtmosWaveCNNTransformer(
        sequence_length=sequence_length,
        mslp_shape=mslp_shape,
        wind_shape=wind_shape,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_layers=hp["num_layers"],
        d_ff=hp["d_ff"],
        dropout=hp["dropout"],
        base_channels=args.base_channels,
        pooled_size=args.pooled_size,
        output_dim=target.shape[1],
    ).to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"device: {device}", flush=True)
    print(f"mslp grid: {mslp_shape}", flush=True)
    print(f"wind grid: {wind_shape}", flush=True)
    print(f"trainable parameters: {n_parameters:,}", flush=True)
    print(f"parameters (million): {n_parameters / 1e6:.3f}", flush=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=hp["l_rate"],
        weight_decay=hp["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=2e-2,
        steps_per_epoch=len(train_loader),
        epochs=hp["max_epochs"],
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_losses = []
    validation_losses = []
    started = time.perf_counter()

    for epoch in range(hp["max_epochs"]):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x1, x2, x3, batch_target in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            prediction = model(x1, x2, x3)
            loss = criterion(prediction, batch_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            count = batch_target.shape[0]
            total_loss += loss.item() * count
            total_count += count

        train_loss = total_loss / total_count
        validation_loss = validate(
            model, validation_loader, criterion, device
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"epoch {epoch + 1:03d} | train {train_loss:.4f} | "
            f"val {validation_loss:.4f}",
            flush=True,
        )

        if validation_loss < best_loss - hp["patience_limit"]:
            best_loss = validation_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hp["patience"]:
                print("early stopping triggered", flush=True)
                break

    if best_state is None:
        raise RuntimeError("no best model state was recorded")

    model.load_state_dict(best_state["model_state_dict"])
    elapsed_seconds = time.perf_counter() - started
    today = datetime.now().strftime("%Y_%m_%d")
    model_path = args.models_dir / (
        f"{today}_atmoswave_cnn_transformer_{args.poi}_"
        f"w{args.window:02d}_sd{args.seed:04d}.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "AtmosWaveCNNTransformer",
            "poi": args.poi,
            "window_hours": args.window,
            "seed": args.seed,
            "grid_shapes": {
                "mslp": mslp_shape,
                "wind": wind_shape,
            },
            "hyperparams": {
                **hp,
                "base_channels": args.base_channels,
                "pooled_size": args.pooled_size,
                "output_dim": target.shape[1],
            },
            "normalisation": {
                "y_mean": y_mean.cpu(),
                "y_std": y_std.cpu(),
            },
            "trainable_parameters": n_parameters,
            "best_epoch": best_state["epoch"] + 1,
            "best_validation_loss": best_loss,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "training_seconds": elapsed_seconds,
            "host": socket.gethostname(),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        },
        model_path,
    )

    print(f"restored best epoch: {best_state['epoch'] + 1}", flush=True)
    print(f"saved checkpoint: {model_path}", flush=True)


if __name__ == "__main__":
    main()
