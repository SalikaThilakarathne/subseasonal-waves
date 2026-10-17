#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# predict wave variables with the spatial cnn-temporal transformer
# salika thilakarathne
# helmholtz-zentrum hereon
# october 2026
# ---------------------------------------------------------------------------

"""Predict the 2008-2023 test period using a trained CNN-Transformer.

Expected test tensors:
    wave_mslp_wind_tensors_SEE_w06_test_2008_2023.pt
    wave_mslp_wind_tensors_SEE_w12_test_2008_2023.pt
    wave_mslp_wind_tensors_SEE_w24_test_2008_2023.pt

Example:
    python -u predict_atmoswave_cnn_transformer.py \
        --poi SEE --window 24 --seed 42 --date 2026_10_17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_atmoswave_cnn_transformer import AtmosWaveCNNTransformer


TEST_START = pd.Timestamp("2008-01-01 00:00:00")
TEST_END = pd.Timestamp("2023-12-31 23:00:00")


def parse_arguments() -> argparse.Namespace:
    """Read the experiment identifier and file locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poi",
        choices=("SEE", "ELB"),
        default="SEE",
    )
    parser.add_argument(
        "--window",
        type=int,
        choices=(6, 12, 24),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="training date used in the checkpoint filename (yyyy_mm_dd)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("../models"),
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("../data/prepared_tensors"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../results"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="inference batch size",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional exact checkpoint path",
    )
    return parser.parse_args()


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    requested_poi: str,
    requested_window: int,
    requested_seed: int,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    """Restore the CNN-Transformer and verify its experiment metadata."""
    # this checkpoint is trusted because it was produced by this workflow
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_class") != "AtmosWaveCNNTransformer":
        raise ValueError(
            "the selected checkpoint is not an AtmosWaveCNNTransformer"
        )

    if checkpoint.get("poi") != requested_poi:
        raise ValueError(
            f"requested poi {requested_poi}, but checkpoint contains "
            f"{checkpoint.get('poi')}"
        )

    if int(checkpoint.get("window_hours", -1)) != requested_window:
        raise ValueError(
            f"requested window {requested_window}, but checkpoint contains "
            f"{checkpoint.get('window_hours')}"
        )

    if int(checkpoint.get("seed", -1)) != requested_seed:
        raise ValueError(
            f"requested seed {requested_seed}, but checkpoint contains "
            f"{checkpoint.get('seed')}"
        )

    hyperparameters = checkpoint["hyperparams"]
    grid_shapes = checkpoint["grid_shapes"]
    mslp_shape = tuple(int(value) for value in grid_shapes["mslp"])
    wind_shape = tuple(int(value) for value in grid_shapes["wind"])

    model = AtmosWaveCNNTransformer(
        sequence_length=requested_window,
        mslp_shape=mslp_shape,
        wind_shape=wind_shape,
        d_model=int(hyperparameters["d_model"]),
        nhead=int(hyperparameters["nhead"]),
        num_layers=int(hyperparameters["num_layers"]),
        d_ff=int(hyperparameters["d_ff"]),
        dropout=float(hyperparameters["dropout"]),
        base_channels=int(hyperparameters["base_channels"]),
        pooled_size=int(hyperparameters["pooled_size"]),
        output_dim=int(hyperparameters["output_dim"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_mean = checkpoint["normalisation"]["y_mean"].to(device)
    y_std = checkpoint["normalisation"]["y_std"].to(device)

    return model, y_mean, y_std, checkpoint


def predict_wave(
    model: nn.Module,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x_mslp: torch.Tensor,
    x_wind_u: torch.Tensor,
    x_wind_v: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run batched inference and return predictions in physical units."""
    predictions = []
    n_samples = x_mslp.shape[0]

    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)

            prediction_normalised = model(
                x_mslp[start:stop].to(device, non_blocking=True),
                x_wind_u[start:stop].to(device, non_blocking=True),
                x_wind_v[start:stop].to(device, non_blocking=True),
            )
            predictions.append(
                (prediction_normalised * y_std + y_mean).cpu()
            )

            if stop == n_samples or stop % (10 * batch_size) == 0:
                print(
                    f"predicted {stop:,} / {n_samples:,} test samples",
                    flush=True,
                )

    return torch.cat(predictions, dim=0)


def main() -> None:
    """Load one checkpoint and predict its matching test-only tensor."""
    args = parse_arguments()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tensor_path = args.tensor_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_"
        f"w{args.window:02d}_test_2008_2023.pt"
    )

    model_path = args.checkpoint or args.models_dir / (
        f"{args.date}_atmoswave_cnn_transformer_{args.poi}_"
        f"w{args.window:02d}_sd{args.seed:04d}.pt"
    )

    if not tensor_path.is_file():
        raise FileNotFoundError(
            f"test tensor file not found: {tensor_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"model checkpoint not found: {model_path}"
        )

    print("=" * 72, flush=True)
    print("cnn-transformer prediction", flush=True)
    print("=" * 72, flush=True)
    print(f"poi: {args.poi}", flush=True)
    print(f"window: {args.window} h", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"date: {args.date}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"model: {model_path}", flush=True)
    print(f"test tensor: {tensor_path}", flush=True)

    model, y_mean, y_std, model_checkpoint = load_model_checkpoint(
        checkpoint_path=model_path,
        device=device,
        requested_poi=args.poi,
        requested_window=args.window,
        requested_seed=args.seed,
    )

    # this file contains only the independent 2008-2023 test period
    tensor_checkpoint = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    x_mslp = tensor_checkpoint["X_mslp"]
    x_wind_u = tensor_checkpoint["X_windU"]
    x_wind_v = tensor_checkpoint["X_windV"]
    y = tensor_checkpoint["y"]
    test_dates = pd.DatetimeIndex(
        pd.to_datetime(tensor_checkpoint["y_dates"])
    )

    if not (
        x_mslp.shape[0]
        == x_wind_u.shape[0]
        == x_wind_v.shape[0]
        == y.shape[0]
        == len(test_dates)
    ):
        raise ValueError(
            "input, target, and date sample counts do not match"
        )

    if not (
        x_mslp.shape[1]
        == x_wind_u.shape[1]
        == x_wind_v.shape[1]
        == args.window
    ):
        raise ValueError(
            "tensor sequence length does not match --window"
        )

    mslp_shape = tuple(model_checkpoint["grid_shapes"]["mslp"])
    wind_shape = tuple(model_checkpoint["grid_shapes"]["wind"])

    if int(np.prod(mslp_shape)) != x_mslp.shape[2]:
        raise ValueError(
            "MSLP grid stored in the checkpoint does not match the tensor"
        )

    if int(np.prod(wind_shape)) != x_wind_u.shape[2]:
        raise ValueError(
            "wind grid stored in the checkpoint does not match the tensor"
        )

    if not test_dates.is_monotonic_increasing:
        raise ValueError(
            "test dates must be sorted in increasing time order"
        )

    if test_dates[0] < TEST_START or test_dates[-1] > TEST_END:
        raise ValueError(
            "test tensor contains timestamps outside 2008-2023"
        )

    print("test tensor shapes:", flush=True)
    print(f"  X_mslp : {tuple(x_mslp.shape)}", flush=True)
    print(f"  X_windU: {tuple(x_wind_u.shape)}", flush=True)
    print(f"  X_windV: {tuple(x_wind_v.shape)}", flush=True)
    print(f"  y      : {tuple(y.shape)}", flush=True)
    print(f"MSLP grid: {mslp_shape}", flush=True)
    print(f"wind grid: {wind_shape}", flush=True)
    print(f"test samples: {len(test_dates):,}", flush=True)
    print(f"test start: {test_dates[0]}", flush=True)
    print(f"test end: {test_dates[-1]}", flush=True)

    predictions = predict_wave(
        model=model,
        y_mean=y_mean,
        y_std=y_std,
        x_mslp=x_mslp,
        x_wind_u=x_wind_u,
        x_wind_v=x_wind_v,
        batch_size=args.batch_size,
        device=device,
    )

    predicted_values = predictions.numpy()
    observed_values = y.numpy()

    output = pd.DataFrame(
        {
            "hs_obs": observed_values[:, 0],
            "t0_obs": observed_values[:, 1],
            "thq_sin_obs": observed_values[:, 2],
            "thq_cos_obs": observed_values[:, 3],
            "hs_pred": predicted_values[:, 0],
            "t0_pred": predicted_values[:, 1],
            "thq_sin_pred": predicted_values[:, 2],
            "thq_cos_pred": predicted_values[:, 3],
        },
        index=test_dates,
    )
    output.index.name = "time"

    # reconstruct wave direction from circular components
    output["thq_rad_obs"] = np.mod(
        np.arctan2(
            output["thq_sin_obs"],
            output["thq_cos_obs"],
        ),
        2 * np.pi,
    )
    output["thq_deg_obs"] = np.degrees(
        output["thq_rad_obs"]
    )
    output["thq_rad_pred"] = np.mod(
        np.arctan2(
            output["thq_sin_pred"],
            output["thq_cos_pred"],
        ),
        2 * np.pi,
    )
    output["thq_deg_pred"] = np.degrees(
        output["thq_rad_pred"]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = args.output_dir / (
        f"{args.date}_atmoswaveCNNTransformer_predictions_{args.poi}_"
        f"w{args.window:02d}_sd{args.seed:04d}.parquet"
    )
    output.to_parquet(
        output_path,
        compression="zstd",
    )

    print(f"saved: {output_path}", flush=True)
    print("cnn-transformer prediction completed successfully", flush=True)


if __name__ == "__main__":
    main()
