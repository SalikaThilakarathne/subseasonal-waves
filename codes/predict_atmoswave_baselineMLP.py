#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# predict wave variables with a trained atmosphere-wave mlp baseline
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# October 2026
# ---------------------------------------------------------------------------

"""Run window-aware mlp predictions for the exact test period.

The data split is identical to the training workflow:

- train: timestamps before 1997-01-01 00:00:00
- validation: 1997-01-01 00:00:00 to before 2008-01-01 00:00:00
- test: timestamps from 2008-01-01 00:00:00 onward

Examples
--------
python -u predict_atmoswave_baselineMLP.py --poi SEE --window 6 --seed 42 --date '2026_10_14'
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_atmoswave_baselineMLP import AtmosWaveMLP


TEST_START = pd.Timestamp("2008-01-01 00:00:00") 
TEST_END = pd.Timestamp("2023-12-31 23:00:00")


def parse_arguments() -> argparse.Namespace:
    """Read the model identifier and input/output paths."""
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
        help="seed identifier used in the checkpoint filename",
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
        default=512,
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
) -> tuple[
    nn.Module,
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    """Load the mlp and verify its experiment metadata."""
    # this checkpoint is trusted because it was produced by this workflow
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("model_type") != "mlp":
        raise ValueError("the selected checkpoint is not an mlp model")

    checkpoint_poi = checkpoint.get("poi")
    checkpoint_window = int(checkpoint.get("window_hours", -1))
    checkpoint_seed = int(checkpoint.get("seed", -1))

    if checkpoint_poi != requested_poi:
        raise ValueError(
            f"requested poi {requested_poi}, but checkpoint contains "
            f"{checkpoint_poi}"
        )

    if checkpoint_window != requested_window:
        raise ValueError(
            f"requested window {requested_window}, but checkpoint contains "
            f"{checkpoint_window}"
        )

    if checkpoint_seed != requested_seed:
        raise ValueError(
            f"requested seed {requested_seed}, but checkpoint contains "
            f"{checkpoint_seed}"
        )

    input_dimensions = checkpoint["input_dimensions"]
    hyperparameters = checkpoint["hyperparameters"]

    model = AtmosWaveMLP(
        input_dim=int(input_dimensions["input_dim"]),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        num_layers=int(hyperparameters["num_layers"]),
        dropout=float(hyperparameters["dropout"]),
        output_dim=int(input_dimensions["output_dim"]),
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
    """Run batched mlp inference and return physical-unit predictions."""
    predictions = []
    n_samples = x_mslp.shape[0]

    # only one batch is transferred to the gpu at a time
    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)

            prediction_normalised = model(
                x_mslp[start:stop].to(device, non_blocking=True),
                x_wind_u[start:stop].to(device, non_blocking=True),
                x_wind_v[start:stop].to(device, non_blocking=True),
            )

            prediction_physical = (
                prediction_normalised * y_std + y_mean
            )
            predictions.append(prediction_physical.cpu())

            if stop == n_samples or stop % (10 * batch_size) == 0:
                print(
                    f"predicted {stop:,} / {n_samples:,} test samples",
                    flush=True,
                )

    return torch.cat(predictions, dim=0)


def main() -> None:
    """Load one mlp experiment, predict the test period, and save parquet."""
    args = parse_arguments()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tensor_path = args.tensor_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_w{args.window:02d}_test_2008_2023.pt"
    )

    if args.checkpoint is None:
        model_path = args.models_dir / (
            f"{args.date}_atmoswave_baselineMLP_{args.poi}_"
            f"w{args.window:02d}_sd{args.seed:04d}.pt"
        )
    else:
        model_path = args.checkpoint

    if not tensor_path.is_file():
        raise FileNotFoundError(
            f"tensor file not found: {tensor_path}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"model checkpoint not found: {model_path}"
        )

    print("=" * 72, flush=True)
    print("atmosphere-wave mlp prediction", flush=True)
    print("=" * 72, flush=True)
    print(f"poi: {args.poi}", flush=True)
    print(f"window: {args.window} h", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"date: {args.date}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"model: {model_path}", flush=True)
    print(f"tensors: {tensor_path}", flush=True)

    model, y_mean, y_std, model_checkpoint = (
        load_model_checkpoint(
            checkpoint_path=model_path,
            device=device,
            requested_poi=args.poi,
            requested_window=args.window,
            requested_seed=args.seed,
        )
    )

    # this tensor file is trusted because it was produced by this workflow
    tensor_checkpoint = torch.load(
        tensor_path,
        map_location="cpu",
        weights_only=False,
    )

    x_mslp = tensor_checkpoint["X_mslp"]
    x_wind_u = tensor_checkpoint["X_windU"]
    x_wind_v = tensor_checkpoint["X_windV"]
    y = tensor_checkpoint["y"]
    dates = pd.DatetimeIndex(
        pd.to_datetime(tensor_checkpoint["y_dates"])
    )

    sample_counts = {
        x_mslp.shape[0],
        x_wind_u.shape[0],
        x_wind_v.shape[0],
        y.shape[0],
        len(dates),
    }

    if len(sample_counts) != 1:
        raise ValueError(
            "input, target, and date sample counts do not match"
        )

    sequence_lengths = {
        x_mslp.shape[1],
        x_wind_u.shape[1],
        x_wind_v.shape[1],
        args.window,
        int(model_checkpoint["window_hours"]),
    }

    if len(sequence_lengths) != 1:
        raise ValueError(
            "model, tensor, and requested windows do not match"
        )

    expected_input_dim = args.window * (
        x_mslp.shape[2]
        + x_wind_u.shape[2]
        + x_wind_v.shape[2]
    )
    checkpoint_input_dim = int(
        model_checkpoint["input_dimensions"]["input_dim"]
    )

    if expected_input_dim != checkpoint_input_dim:
        raise ValueError(
            f"tensor input dimension is {expected_input_dim}, but model "
            f"expects {checkpoint_input_dim}"
        )

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "y_dates must be sorted in increasing time order"
        )

    # use exactly the same date boundaries as the training script
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "test dates must be sorted in increasing time order"
        )
    
    if dates[0] < TEST_START:
        raise ValueError(
            f"test tensor begins before {TEST_START}: "
            f"{dates[0]}"
        )
    
    if dates[-1] > TEST_END:
        raise ValueError(
            f"test tensor ends after {TEST_END}: "
            f"{dates[-1]}"
        )
    
    test_dates = dates
    
    print("test tensor shapes:", flush=True)
    print(f"  X_mslp : {tuple(x_mslp.shape)}", flush=True)
    print(f"  X_windU: {tuple(x_wind_u.shape)}", flush=True)
    print(f"  X_windV: {tuple(x_wind_v.shape)}", flush=True)
    print(f"  y      : {tuple(y.shape)}", flush=True)
    
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

    # reconstruct wave direction from its circular components
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
        f"{args.date}_atmoswave_bMLP_predictions_{args.poi}_"
        f"w{args.window:02d}_sd{args.seed:04d}.parquet"
    )
    output.to_parquet(
        output_path,
        compression="zstd",
    )

    print(f"saved: {output_path}", flush=True)
    print("mlp prediction completed successfully", flush=True)


if __name__ == "__main__":
    main()
