#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# wave prediction with mslp + wind using a transformer encoder
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# repository: # https://gitlab.hzdr.de/...
# March 2026
# ---------------------------------------------------------------------------

"""Create aligned wave, mean sea-level pressure, and wind tensors.

This script prepares the model inputs used for the transformer history-window
experiments. For a window ending at time ``t``, the atmospheric predictors
cover ``t-window+1`` through ``t`` and the wave target is the state at ``t``.
Thus, this is a concurrent reconstruction experiment rather than a forecast
with a future lead time.

The script performs the following reproducible steps:

1. reads standardized mslp, wind-u, and wind-v predictor tables;
2. reads wave targets for one point of interest;
3. converts wave direction to sine and cosine components;
4. aligns all four sources on their common hourly timestamps;
5. rejects windows that cross a missing hourly timestep;
6. creates float32 tensors with shape (sample, time, feature);
7. saves the tensors, target dates, metadata, and a json manifest.

Input files expected by default
-------------------------------
../data/stdanom_mslp_data.parquet
../data/stdanom_windU_data.parquet
../data/stdanom_windV_data.parquet
../data/prepared_wave/wave_data_processed_SEE.parquet

Examples
--------
python create_wave_tensors_reproducible.py --poi SEE --window 6
python create_wave_tensors_reproducible.py --poi SEE --window 12
python create_wave_tensors_reproducible.py --poi SEE --window 24

Use ``--wave-file`` when the wave file is parquet or has a different name.
Use ``--overwrite`` only when an existing result should be replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# target order is fixed because the training code reads these positions directly
TARGET_COLUMNS = ["hs", "t0", "thq_sin", "thq_cos"]


def log(message: str = "") -> None:
    """Write a timestamped message immediately to the SLURM log."""
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}",
        flush=True,
    )


def tensor_gib(tensor: torch.Tensor) -> float:
    """Return the in-memory size of a tensor in GiB."""
    return tensor.numel() * tensor.element_size() / 1024**3


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poi",
        required=True,
        help="Location code, e.g. SEE, ELB, or NST",
    )
    parser.add_argument(
        "--window",
        required=True,
        type=int,
        choices=(6, 12, 24),
        help="History-window length in hours",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/prepared_tensors"),
    )
    parser.add_argument(
        "--wave-file",
        type=Path,
        default=None,
        help="Optional CSV or Parquet wave file",
    )
    parser.add_argument(
        "--direction-unit",
        choices=("auto", "degrees", "radians"),
        default="auto",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file",
    )
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    """Read a time-indexed CSV or Parquet DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    if path.suffix.lower() in (".parquet", ".pq"):
        dataframe = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    # use one sorted datetime index for deterministic temporal alignment
    dataframe.index = pd.to_datetime(dataframe.index)

    # duplicate times would make the definition of a history window ambiguous
    if dataframe.index.has_duplicates:
        examples = (
            dataframe.index[dataframe.index.duplicated()]
            .unique()[:5]
            .tolist()
        )
        raise ValueError(
            f"Duplicate timestamps in {path}; examples: {examples}"
        )

    return dataframe.sort_index()


def prepare_wave(
    wave: pd.DataFrame,
    direction_unit: str,
) -> pd.DataFrame:
    """Prepare hs, t0, and sine/cosine wave-direction targets."""
    wave = wave.copy()

    if "tm2" in wave.columns and "t0" not in wave.columns:
        wave.rename(
            columns={"tm2": "t0"},
            inplace=True,
        )

    if "delta" in wave.columns and "thq" not in wave.columns:
        wave.rename(
            columns={"delta": "thq"},
            inplace=True,
        )

    required = {"hs", "t0", "thq"}
    missing = required.difference(wave.columns)

    if missing:
        raise KeyError(
            f"Missing wave columns: {sorted(missing)}"
        )

    # preserve the original workflow: replace missing t0 only
    n_missing_t0 = int(wave["t0"].isna().sum())

    if n_missing_t0:
        minimum_t0 = wave["t0"].min(skipna=True)

        if not np.isfinite(minimum_t0):
            raise ValueError(
                "t0 contains no finite values"
            )

        wave["t0"] = wave["t0"].fillna(minimum_t0)
        log(
            f"Filled {n_missing_t0} missing t0 values "
            f"with {minimum_t0}."
        )

    # encode circular direction as two continuous regression targets
    theta = wave["thq"].to_numpy(dtype=np.float64)
    appears_to_be_degrees = (
        direction_unit == "auto"
        and np.nanmax(np.abs(theta)) > 2 * np.pi + 1e-3
    )

    if direction_unit == "degrees" or appears_to_be_degrees:
        theta = np.deg2rad(theta)

    wave["thq_sin"] = np.sin(theta)
    wave["thq_cos"] = np.cos(theta)

    return wave[TARGET_COLUMNS]


def continuous_endpoint_mask(
    index: pd.DatetimeIndex,
    window: int,
) -> np.ndarray:
    """Return one flag per candidate window endpoint."""
    n_candidates = len(index) - window + 1

    if n_candidates <= 0:
        return np.zeros(0, dtype=bool)

    if window == 1:
        return np.ones(n_candidates, dtype=bool)

    # a valid sample must contain only consecutive one-hour time steps
    one_hour = pd.Timedelta(hours=1).value
    hourly_steps = np.diff(index.asi8) == one_hour

    return np.lib.stride_tricks.sliding_window_view(
        hourly_steps,
        window - 1,
    ).all(axis=1)


def make_tensor(
    dataframe: pd.DataFrame,
    window: int,
    keep: np.ndarray,
) -> torch.Tensor:
    """Build a contiguous float32 tensor: sample x time x feature."""
    # float32 matches model training and halves memory relative to float64
    values = np.ascontiguousarray(
        dataframe.to_numpy(
            dtype=np.float32,
            copy=True,
        )
    )

    # unfold creates all overlapping windows before invalid endpoints are selected
    all_windows = (
        torch.from_numpy(values)
        .unfold(0, window, 1)
        .permute(0, 2, 1)
    )

    selected_indices = torch.from_numpy(
        np.flatnonzero(keep)
    ).long()

    return all_windows.index_select(
        0,
        selected_indices,
    ).contiguous()


def check_predictors(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Stop if a predictor DataFrame contains NaN or infinite values."""
    values = dataframe.to_numpy(copy=False)
    # fail early because nan or infinity values would invalidate model training
    finite = np.isfinite(values)

    if not finite.all():
        affected_columns = (
            dataframe.columns[(~finite).any(axis=0)]
            .astype(str)
            .tolist()[:10]
        )
        n_bad = int((~finite).sum())

        raise ValueError(
            f"{name} has {n_bad} non-finite values; "
            f"affected columns include {affected_columns}"
        )


def main() -> None:
    started = time.perf_counter()
    args = parse_arguments()

    wave_path = ( 
        args.wave_file 
        or args.data_dir / "prepared_wave" / 
        "wave_data_processed_SEE.parquet" 
    )

    # predictor files are shared across points of interest
    predictor_paths = {
        "MSLP": args.data_dir / "stdanom_mslp_data.parquet",
        "WindU": args.data_dir / "stdanom_windU_data.parquet",
        "WindV": args.data_dir / "stdanom_windV_data.parquet",
    }

    log("=" * 72)
    log("WAVE / MSLP / WIND TENSOR PREPARATION")
    log("=" * 72)
    log(
        f"Host: {socket.gethostname()} | "
        f"Job: {os.getenv('SLURM_JOB_ID', 'non-SLURM')} | "
        f"Array: {os.getenv('SLURM_ARRAY_TASK_ID', 'n/a')}"
    )
    log(
        f"POI={args.poi} | "
        f"window={args.window} h | "
        f"output={args.output_dir}"
    )
    log(f"Wave={wave_path}")

    for name, path in predictor_paths.items():
        log(f"{name}={path}")

    loading_started = time.perf_counter()

    log("Loading wave data...")
    wave = prepare_wave(
        read_frame(wave_path),
        args.direction_unit,
    )
    log(f"Wave: {wave.shape}")

    log("Loading MSLP...")
    mslp = read_frame(predictor_paths["MSLP"])
    log(f"MSLP: {mslp.shape}")

    log("Loading wind-U...")
    wind_u = read_frame(predictor_paths["WindU"])
    log(f"WindU: {wind_u.shape}")

    log("Loading wind-V...")
    wind_v = read_frame(predictor_paths["WindV"])
    log(f"WindV: {wind_v.shape}")

    loading_minutes = (
        time.perf_counter() - loading_started
    ) / 60
    log(f"Loading completed in {loading_minutes:.2f} min")

    # align before window construction so every source refers to the same hour
    log("Aligning timestamps...")
    common_index = (
        wave.index
        .intersection(mslp.index)
        .intersection(wind_u.index)
        .intersection(wind_v.index)
        .sort_values()
    )

    if common_index.empty:
        raise ValueError(
            "The four inputs have no common timestamps"
        )

    wave = wave.loc[common_index]
    mslp = mslp.loc[common_index]
    wind_u = wind_u.loc[common_index]
    wind_v = wind_v.loc[common_index]

    log(
        f"Aligned {len(common_index):,} rows: "
        f"{common_index[0]} to {common_index[-1]}"
    )

    predictors = (
        ("MSLP", mslp),
        ("WindU", wind_u),
        ("WindV", wind_v),
    )

    for name, dataframe in predictors:
        log(f"Checking {name} finite values...")
        check_predictors(name, dataframe)
        log(f"{name} check passed")

    # keep only endpoints whose full atmospheric history is continuous
    keep = continuous_endpoint_mask(
        common_index,
        args.window,
    )

    endpoint_targets = wave.iloc[
        args.window - 1:
    ].to_numpy(dtype=np.float32)

    n_continuous = int(keep.sum())
    # remove samples with an invalid target while retaining the same predictor mask
    finite_targets = np.isfinite(endpoint_targets).all(axis=1)
    keep &= finite_targets
    n_samples = int(keep.sum())

    if n_samples == 0:
        raise ValueError(
            "No valid samples remain"
        )

    n_features = (
        mslp.shape[1]
        + wind_u.shape[1]
        + wind_v.shape[1]
    )
    # estimate final tensor payload; peak ram is higher because temporary copies exist
    estimated_bytes = (
        n_samples * args.window * n_features * 4
        + n_samples * len(TARGET_COLUMNS) * 4
    )
    estimated_gib = estimated_bytes / 1024**3

    log(
        f"Candidates={len(keep):,} | "
        f"continuous={n_continuous:,} | "
        f"final={n_samples:,}"
    )
    log(
        f"Features: MSLP={mslp.shape[1]:,}, "
        f"WindU={wind_u.shape[1]:,}, "
        f"WindV={wind_v.shape[1]:,}"
    )
    log(
        f"Estimated payload={estimated_gib:.2f} GiB "
        "plus temporary memory"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = args.output_dir / (
        f"wave_mslp_wind_tensors_{args.poi}_"
        f"w{args.window:02d}.pt"
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists; use --overwrite"
        )

    # build one predictor tensor at a time and report progress to the slurm log
    tensors = {}

    for key, dataframe in (
        ("X_mslp", mslp),
        ("X_windU", wind_u),
        ("X_windV", wind_v),
    ):
        tensor_started = time.perf_counter()
        log(f"Building {key}...")

        tensors[key] = make_tensor(
            dataframe,
            args.window,
            keep,
        )

        tensor_minutes = (
            time.perf_counter() - tensor_started
        ) / 60
        log(
            f"{key} complete | "
            f"shape={tuple(tensors[key].shape)} | "
            f"size={tensor_gib(tensors[key]):.2f} GiB | "
            f"time={tensor_minutes:.2f} min"
        )

    # targets and timestamps correspond to the final hour of each retained window
    y = torch.from_numpy(
        endpoint_targets[keep].copy()
    )
    y_dates = (
        common_index[args.window - 1:][keep]
        .astype("datetime64[ns]")
        .astype(str)
        .tolist()
    )

    n_tensor_samples = {
        tensors["X_mslp"].shape[0],
        tensors["X_windU"].shape[0],
        tensors["X_windV"].shape[0],
        y.shape[0],
        len(y_dates),
    }

    if len(n_tensor_samples) != 1:
        raise RuntimeError(
            "Predictor, target, and date sample counts do not match"
        )

    # embed provenance needed to interpret the tensor file independently
    metadata = {
        "poi": args.poi,
        "window_hours": args.window,
        "window_definition": (
            "predictors t-window+1 through t; target at t"
        ),
        "target_columns": TARGET_COLUMNS,
        "n_aligned_rows": len(common_index),
        "n_samples": n_samples,
        "aligned_start": str(common_index[0]),
        "aligned_end": str(common_index[-1]),
        "mslp_features": list(map(str, mslp.columns)),
        "windU_features": list(map(str, wind_u.columns)),
        "windV_features": list(map(str, wind_v.columns)),
    }

    # keep key names compatible with the transformer training workflow
    checkpoint = {
        **tensors,
        "y": y,
        "y_dates": y_dates,
        "metadata": metadata,
    }

    log(f"Saving checkpoint to {output_path}...")
    saving_started = time.perf_counter()
    torch.save(checkpoint, output_path)
    saving_minutes = (
        time.perf_counter() - saving_started
    ) / 60
    log(f"Checkpoint saved in {saving_minutes:.2f} min")

    # write a small human-readable summary without reopening the large tensor file
    manifest_path = output_path.with_suffix(".json")
    manifest = {
        "tensor_file": output_path.name,
        "poi": args.poi,
        "window_hours": args.window,
        "n_samples": n_samples,
        "shapes": {
            key: list(value.shape)
            for key, value in {**tensors, "y": y}.items()
        },
        "sizes_gib": {
            key: tensor_gib(value)
            for key, value in {**tensors, "y": y}.items()
        },
        "estimated_payload_gib": estimated_gib,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2)
    )

    total_minutes = (
        time.perf_counter() - started
    ) / 60
    log(f"Manifest saved: {manifest_path}")
    log(f"Total elapsed: {total_minutes:.2f} min")
    log(
        f"[{args.poi}] [{args.window}h] "
        "Tensor generation successful."
    )
    log("=" * 72)


if __name__ == "__main__":
    main()