#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# train an mlp baseline for atmosphere-wave prediction
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# October 2026
# ---------------------------------------------------------------------------

"""
Train a multilayer perceptron baseline using prepared atmosphere-wave tensors-- based on the reviewer comments, this part is added to the experiment design!!-- The mlp uses the same tensor inputs, target standardisation, date-based split, optimiser settings, loss function, and early-stopping settings as the transformer experiments. the temporal and feature dimensions are flattened before the mlp layers; therefore, the model has no attention mechanism.

examples -- run as slurm jobs
--------
python -u train_atmoswave_baselineMLP.py --poi SEE --window 06 --seed 42
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
TARGET_COLUMNS = ("hs", "t0", "thq_sin", "thq_cos")


class AtmosWaveMLP(nn.Module):
    """Predict four wave variables from "flattened" atmospheric histories."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 4,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers: list[nn.Module] = []
        in_features = input_dim

        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x_mslp: torch.Tensor,
        x_wind_u: torch.Tensor,
        x_wind_v: torch.Tensor,
    ) -> torch.Tensor:
        # flatten time and space, then concatenate all atmospheric predictors
        batch_size = x_mslp.shape[0]
        x = torch.cat(
            (
                x_mslp.reshape(batch_size, -1),
                x_wind_u.reshape(batch_size, -1),
                x_wind_v.reshape(batch_size, -1),
            ),
            dim=1,
        )
        return self.network(x)


def parse_arguments() -> argparse.Namespace:
    """Read experiment identifiers and file locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poi",
        choices=("SEE", "ELB"),
        default="SEE",
        help="SEE is the main experiment; ELB is the transferability check",
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=(6, 12, 24),
        required=True,
        help="atmospheric history length in hours",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for model initialisation and data shuffling -- uncertainty quantification!",
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("../data/prepared_tensors"),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("../models"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("../config/mlp_hyperparams.txt"),
        help="optional key=value hyperparameter file",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    """Set python, numpy, and pytorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_hyperparameters(path: Path) -> dict:
    """Load key=value settings and apply publication defaults--similar values as our Transformer archi"""
    hyperparameters = {
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1,
        "l_rate": 0.001,
        "weight_decay": 0.001,
        "batch_size": 1024,
        "max_epochs": 500,
        "patience": 50,
        "patience_limit": 0.0005,
    }

    if not path.is_file():
        print(f"config not found; using defaults: {path}", flush=True)
        return hyperparameters

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, value = (part.strip() for part in line.split("=", 1))
        if key not in hyperparameters:
            raise KeyError(f"unsupported mlp hyperparameter: {key}")

        current = hyperparameters[key]
        hyperparameters[key] = type(current)(value)

    return hyperparameters


def make_loader(
    x_mslp: torch.Tensor,
    x_wind_u: torch.Tensor,
    x_wind_v: torch.Tensor,
    y: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a data loader from one date-defined subset."""
    index_tensor = torch.from_numpy(indices).long()
    dataset = TensorDataset(
        x_mslp.index_select(0, index_tensor),
        x_wind_u.index_select(0, index_tensor),
        x_wind_v.index_select(0, index_tensor),
        y.index_select(0, index_tensor),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return sample-weighted mean loss for one data loader."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x_mslp, x_wind_u, x_wind_v, target in loader:
            prediction = model(
                x_mslp.to(device, non_blocking=True),
                x_wind_u.to(device, non_blocking=True),
                x_wind_v.to(device, non_blocking=True),
            )
            target = target.to(device, non_blocking=True)
            loss = criterion(prediction, target)
            batch_count = target.shape[0]
            total_loss += loss.item() * batch_count
            total_count += batch_count

    return total_loss / total_count


def main() -> None:
    """Load tensors, train the mlp, and save the best validation model."""
    args = parse_arguments()
    set_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor_path = args.tensor_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_w{args.window:02d}.pt"
    )
    if not tensor_path.is_file():
        raise FileNotFoundError(f"tensor file not found: {tensor_path}")

    hyperparameters = load_hyperparameters(args.config)

    print("=" * 72, flush=True)
    print("atmosphere-wave mlp training", flush=True)
    print("=" * 72, flush=True)
    print(f"host: {socket.gethostname()}", flush=True)
    print(f"slurm job: {os.getenv('SLURM_JOB_ID', 'non-slurm')}", flush=True)
    print(f"poi: {args.poi}", flush=True)
    print(f"window: {args.window} h", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"loading tensors from: {tensor_path}", flush=True)

    # the tensor file is trusted because it is produced by this workflow
    checkpoint = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )
    x_mslp = checkpoint["X_mslp"]
    x_wind_u = checkpoint["X_windU"]
    x_wind_v = checkpoint["X_windV"]
    y = checkpoint["y"]
    dates = pd.DatetimeIndex(pd.to_datetime(checkpoint["y_dates"]))

    if not (
        x_mslp.shape[0]
        == x_wind_u.shape[0]
        == x_wind_v.shape[0]
        == y.shape[0]
        == len(dates)
    ):
        raise ValueError("input, target, and date sample counts do not match")

    if not (
        x_mslp.shape[1]
        == x_wind_u.shape[1]
        == x_wind_v.shape[1]
        == args.window
    ):
        raise ValueError("tensor sequence length does not match --window length is checked! 6/12/24")

    train_indices = np.flatnonzero(dates < TRAIN_END)
    validation_indices = np.flatnonzero(
        (dates >= TRAIN_END) & (dates < VALIDATION_END)
    )
    test_indices = np.flatnonzero(dates >= VALIDATION_END)

    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("an empty train, validation, or test period was found")

    # target statistics are calculated from the training period only
    y_train = y.index_select(0, torch.from_numpy(train_indices).long())
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0)
    if torch.any(y_std == 0):
        raise ValueError("at least one target has zero training-period standard deviation")
    y_normalised = (y - y_mean) / y_std

    batch_size = int(hyperparameters["batch_size"])
    train_loader = make_loader(
        x_mslp,
        x_wind_u,
        x_wind_v,
        y_normalised,
        train_indices,
        batch_size,
        True,
        args.seed,
    )
    validation_loader = make_loader(
        x_mslp,
        x_wind_u,
        x_wind_v,
        y_normalised,
        validation_indices,
        2048,
        False,
        args.seed,
    )

    input_dim = args.window * (
        x_mslp.shape[2] + x_wind_u.shape[2] + x_wind_v.shape[2]
    )
    model = AtmosWaveMLP(
        input_dim=input_dim,
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        dropout=float(hyperparameters["dropout"]),
        output_dim=y.shape[1],
    ).to(device)

    # use the same loss, optimiser, and learning-rate schedule as the transformer
    criterion = nn.HuberLoss(delta=1.0)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["l_rate"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=2e-2,
        steps_per_epoch=len(train_loader),
        epochs=int(hyperparameters["max_epochs"]),
    )

    print("tensor shapes:", flush=True)
    print(f"  X_mslp : {tuple(x_mslp.shape)}", flush=True)
    print(f"  X_windU: {tuple(x_wind_u.shape)}", flush=True)
    print(f"  X_windV: {tuple(x_wind_v.shape)}", flush=True)
    print(f"  y      : {tuple(y.shape)}", flush=True)
    print("date-based split:", flush=True)
    print(f"  train      : {len(train_indices):,}", flush=True)
    print(f"  validation : {len(validation_indices):,}", flush=True)
    print(f"  test       : {len(test_indices):,}", flush=True)
    print(f"mlp input dimension: {input_dim:,}", flush=True)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("hyperparameters:", flush=True)
    for key, value in hyperparameters.items():
        print(f"  {key:16s} = {value}", flush=True)
    print("training controls matched to transformer:", flush=True)
    print("  loss             = HuberLoss(delta=1.0)", flush=True)
    print("  optimiser        = AdamW", flush=True)
    print("  scheduler        = OneCycleLR", flush=True)
    print("  scheduler max_lr = 0.02", flush=True)
    print(f"  train batch      = {batch_size}", flush=True)
    print("  validation batch = 2048", flush=True)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    validation_losses = []
    started = time.perf_counter()

    for epoch in range(int(hyperparameters["max_epochs"])):
        model.train()
        total_loss = 0.0
        total_count = 0

        for x1, x2, x3, target in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            prediction = model(x1, x2, x3)
            loss = criterion(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            batch_count = target.shape[0]
            total_loss += loss.item() * batch_count
            total_count += batch_count

        train_loss = total_loss / total_count
        validation_loss = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(
            f"epoch {epoch + 1:03d} | train {train_loss:.4f} | "
            f"val {validation_loss:.4f}",
            flush=True,
        )

        minimum_change = float(hyperparameters["patience_limit"])
        if validation_loss < best_val_loss - minimum_change:
            best_val_loss = validation_loss
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(hyperparameters["patience"]):
                print("early stopping triggered", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training ended without a best model state")

    model.load_state_dict(best_state["model_state_dict"])
    elapsed_seconds = time.perf_counter() - started
    args.models_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y_%m_%d")
    stem = (
        f"{today}_atmoswave_baselineMLP_{args.poi}_"
        f"w{args.window:02d}_sd{args.seed:04d}"
    )
    model_path = args.models_dir / f"{stem}.pt"
    history_path = args.models_dir / f"{stem}_history.json"

    output_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "mlp",
        "poi": args.poi,
        "window_hours": args.window,
        "seed": args.seed,
        "experiment_role": (
            "mlp baseline at primary site"
            if args.poi == "SEE"
            else "mlp transferability check"
        ),
        "best_epoch": best_state["epoch"] + 1,
        "best_validation_loss": best_val_loss,
        "input_dimensions": {
            "T": args.window,
            "F1": x_mslp.shape[2],
            "F2": x_wind_u.shape[2],
            "F3": x_wind_v.shape[2],
            "input_dim": input_dim,
            "output_dim": y.shape[1],
        },
        "hyperparameters": hyperparameters,
        "training_controls": {
            "loss": "HuberLoss",
            "huber_delta": 1.0,
            "optimiser": "AdamW",
            "scheduler": "OneCycleLR",
            "scheduler_max_lr": 2e-2,
            "train_batch_size": batch_size,
            "validation_batch_size": 2048,
            "gradient_clip_norm": 1.0,
        },
        "normalisation": {
            "y_mean": y_mean,
            "y_std": y_std,
        },
        "target_columns": TARGET_COLUMNS,
        "date_split": {
            "train_end_exclusive": str(TRAIN_END),
            "validation_end_exclusive": str(VALIDATION_END),
        },
        "training_seconds": elapsed_seconds,
    }
    torch.save(output_checkpoint, model_path)

    history_path.write_text(
        json.dumps(
            {
                "model_file": model_path.name,
                "poi": args.poi,
                "window_hours": args.window,
                "seed": args.seed,
                "best_epoch": best_state["epoch"] + 1,
                "best_validation_loss": best_val_loss,
                "training_seconds": elapsed_seconds,
                "train_loss": train_losses,
                "validation_loss": validation_losses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"restored best model from epoch {best_state['epoch'] + 1}", flush=True)
    print(f"best validation loss: {best_val_loss:.6f}", flush=True)
    print(f"training time: {elapsed_seconds / 60:.2f} min", flush=True)
    print(f"saved model: {model_path}", flush=True)
    print(f"saved history: {history_path}", flush=True)


if __name__ == "__main__":
    main()