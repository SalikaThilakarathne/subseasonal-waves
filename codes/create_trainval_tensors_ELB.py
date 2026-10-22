#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# create train-validation atmosphere-wave tensors for see
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# October 2026
# ---------------------------------------------------------------------------

"""Create 6 h, 12 h, and 24 h tensors for the 1948-2007 period.

The saved files contain only targets before 1 january 2008. they are intended
for chronological model development using:

- training: timestamps before 1 january 1997
- validation: 1 january 1997 to 31 december 2007

Outputs
-------
wave_mslp_wind_tensors_SEE_w06_trainval_1948_2007.pt
wave_mslp_wind_tensors_SEE_w12_trainval_1948_2007.pt
wave_mslp_wind_tensors_SEE_w24_trainval_1948_2007.pt

Example
-------
python -u create_trainval_tensors_SEE.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


POI = "ELB"
TRAINVAL_START = pd.Timestamp("1948-01-01 00:00:00")
TRAIN_END = pd.Timestamp("1997-01-01 00:00:00")
TRAINVAL_END = pd.Timestamp("2007-12-31 23:00:00")
TEST_START = pd.Timestamp("2008-01-01 00:00:00")
TARGET_COLUMNS = ["hs", "t0", "thq_sin", "thq_cos"]


def parse_arguments() -> argparse.Namespace:
    """Read source locations and requested history windows."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--windows",
        type=int,
        nargs="+",
        choices=(6, 12, 24),
        default=(6, 12, 24),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def read_parquet_time_index(path: Path) -> pd.DataFrame:
    """Read one parquet file and enforce a sorted datetime index."""
    if not path.is_file():
        raise FileNotFoundError(f"input file not found: {path}")

    print(f"loading: {path}", flush=True)
    dataframe = pd.read_parquet(path)
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.sort_index(inplace=True)

    if dataframe.index.has_duplicates:
        raise ValueError(f"duplicate timestamps found in: {path}")

    print(
        f"loaded shape={dataframe.shape}, "
        f"period={dataframe.index[0]} to {dataframe.index[-1]}",
        flush=True,
    )
    return dataframe


def check_wave_targets(wave: pd.DataFrame) -> pd.DataFrame:
    """Validate and select the four wave targets."""
    missing = set(TARGET_COLUMNS).difference(wave.columns)

    if missing:
        raise KeyError(
            f"prepared wave data are missing columns: {sorted(missing)}"
        )

    return wave[TARGET_COLUMNS].copy()


def check_finite_predictors(name: str, dataframe: pd.DataFrame) -> None:
    """Stop if a predictor contains nan or infinite values."""
    values = dataframe.to_numpy(copy=False)
    n_bad = int((~np.isfinite(values)).sum())

    if n_bad:
        raise ValueError(
            f"{name} contains {n_bad:,} non-finite predictor values"
        )


def continuous_window_mask(
    index: pd.DatetimeIndex,
    window: int,
) -> np.ndarray:
    """Flag candidate windows whose timestamps are hourly and continuous."""
    n_candidates = len(index) - window + 1

    if n_candidates <= 0:
        return np.zeros(0, dtype=bool)

    hourly_steps = (
        np.diff(index.asi8)
        == pd.Timedelta(hours=1).value
    )

    return np.lib.stride_tricks.sliding_window_view(
        hourly_steps,
        window - 1,
    ).all(axis=1)


def make_tensor(
    dataframe: pd.DataFrame,
    window: int,
    keep: np.ndarray,
) -> torch.Tensor:
    """Create a float32 tensor with shape sample x time x feature."""
    values = np.ascontiguousarray(
        dataframe.to_numpy(dtype=np.float32, copy=True)
    )
    windows = (
        torch.from_numpy(values)
        .unfold(0, window, 1)
        .permute(0, 2, 1)
    )
    selected = torch.from_numpy(
        np.flatnonzero(keep)
    ).long()
    return windows.index_select(0, selected).contiguous()


def tensor_gib(tensor: torch.Tensor) -> float:
    """Return a tensor size in gibibytes."""
    return tensor.numel() * tensor.element_size() / 1024**3


def create_window_file(
    mslp: pd.DataFrame,
    wind_u: pd.DataFrame,
    wind_v: pd.DataFrame,
    wave: pd.DataFrame,
    window: int,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    """Create one train-validation tensor checkpoint."""
    index = wave.index
    continuity = continuous_window_mask(index, window)
    endpoint_dates = index[window - 1:]
    endpoint_targets = wave.iloc[
        window - 1:
    ].to_numpy(dtype=np.float32)

    # retain only finite targets within the train-validation period
    keep = continuity.copy()
    keep &= np.asarray(endpoint_dates >= TRAINVAL_START)
    keep &= np.asarray(endpoint_dates <= TRAINVAL_END)
    keep &= np.isfinite(endpoint_targets).all(axis=1)

    if not keep.any():
        raise ValueError(
            f"no valid {window}-hour train-validation windows were found"
        )

    output_path = output_dir / (
        f"wave_mslp_wind_tensors_{POI}_"
        f"w{window:02d}_trainval_1948_2007.pt"
    )

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; use --overwrite to replace it"
        )

    kept_dates = endpoint_dates[keep]
    n_train = int((kept_dates < TRAIN_END).sum())
    n_validation = int(
        ((kept_dates >= TRAIN_END) & (kept_dates < TEST_START)).sum()
    )
    n_samples = int(keep.sum())
    n_features = (
        mslp.shape[1]
        + wind_u.shape[1]
        + wind_v.shape[1]
    )
    estimated_gib = (
        n_samples * window * n_features * 4
        + n_samples * len(TARGET_COLUMNS) * 4
    ) / 1024**3

    print("=" * 72, flush=True)
    print(
        f"creating {POI} w{window:02d} train-validation tensors",
        flush=True,
    )
    print(f"target start: {kept_dates[0]}", flush=True)
    print(f"target end: {kept_dates[-1]}", flush=True)
    print(f"training samples: {n_train:,}", flush=True)
    print(f"validation samples: {n_validation:,}", flush=True)
    print(f"total samples: {n_samples:,}", flush=True)
    print(
        f"estimated final payload: {estimated_gib:.2f} gib",
        flush=True,
    )

    started = time.perf_counter()

    print("building X_mslp...", flush=True)
    x_mslp = make_tensor(mslp, window, keep)
    print(
        f"X_mslp complete: {tuple(x_mslp.shape)}, "
        f"{tensor_gib(x_mslp):.2f} gib",
        flush=True,
    )

    print("building X_windU...", flush=True)
    x_wind_u = make_tensor(wind_u, window, keep)
    print(
        f"X_windU complete: {tuple(x_wind_u.shape)}, "
        f"{tensor_gib(x_wind_u):.2f} gib",
        flush=True,
    )

    print("building X_windV...", flush=True)
    x_wind_v = make_tensor(wind_v, window, keep)
    print(
        f"X_windV complete: {tuple(x_wind_v.shape)}, "
        f"{tensor_gib(x_wind_v):.2f} gib",
        flush=True,
    )

    y = torch.from_numpy(endpoint_targets[keep].copy())
    y_dates = kept_dates.astype(
        "datetime64[ns]"
    ).astype(str).tolist()

    # strict leakage check before saving
    if pd.DatetimeIndex(pd.to_datetime(y_dates)).max() >= TEST_START:
        raise ValueError(
            "train-validation tensor contains test-period target timestamps"
        )

    checkpoint = {
        "X_mslp": x_mslp,
        "X_windU": x_wind_u,
        "X_windV": x_wind_v,
        "y": y,
        "y_dates": y_dates,
        "metadata": {
            "poi": POI,
            "window_hours": window,
            "dataset_role": "train-validation period",
            "target_start": str(kept_dates[0]),
            "target_end": str(kept_dates[-1]),
            "train_end_exclusive": str(TRAIN_END),
            "test_start_exclusive": str(TEST_START),
            "target_columns": TARGET_COLUMNS,
            "n_train": n_train,
            "n_validation": n_validation,
            "n_samples": n_samples,
            "mslp_features": mslp.shape[1],
            "windU_features": wind_u.shape[1],
            "windV_features": wind_v.shape[1],
        },
    }

    print(f"saving: {output_path}", flush=True)
    torch.save(checkpoint, output_path)

    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(
            {
                "tensor_file": output_path.name,
                "poi": POI,
                "window_hours": window,
                "dataset_role": "train-validation period",
                "target_start": str(kept_dates[0]),
                "target_end": str(kept_dates[-1]),
                "n_train": n_train,
                "n_validation": n_validation,
                "n_samples": n_samples,
                "shapes": {
                    "X_mslp": list(x_mslp.shape),
                    "X_windU": list(x_wind_u.shape),
                    "X_windV": list(x_wind_v.shape),
                    "y": list(y.shape),
                },
                "payload_gib": (
                    tensor_gib(x_mslp)
                    + tensor_gib(x_wind_u)
                    + tensor_gib(x_wind_v)
                    + tensor_gib(y)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed_minutes = (
        time.perf_counter() - started
    ) / 60
    print(f"saved: {output_path}", flush=True)
    print(f"saved: {manifest_path}", flush=True)
    print(f"elapsed: {elapsed_minutes:.2f} min", flush=True)

    # release the large tensors before creating the next window
    del x_mslp, x_wind_u, x_wind_v, y, checkpoint
    gc.collect()

    return output_path


def main() -> None:
    """Load prepared tables once and create all requested window files."""
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "mslp": args.data_dir / "stdanom_mslp_data.parquet",
        "wind_u": args.data_dir / "stdanom_windU_data.parquet",
        "wind_v": args.data_dir / "stdanom_windV_data.parquet",
        "wave": (
            args.data_dir
            / "prepared_wave"
            / "wave_data_processed_SEE.parquet"
        ),
    }

    # use only rows needed for train-validation targets and their history
    maximum_history_start = TRAINVAL_START - pd.Timedelta(hours=23)

    mslp = read_parquet_time_index(paths["mslp"]).loc[
        maximum_history_start:TRAINVAL_END
    ]
    wind_u = read_parquet_time_index(paths["wind_u"]).loc[
        maximum_history_start:TRAINVAL_END
    ]
    wind_v = read_parquet_time_index(paths["wind_v"]).loc[
        maximum_history_start:TRAINVAL_END
    ]
    wave = check_wave_targets(
        read_parquet_time_index(paths["wave"])
    ).loc[maximum_history_start:TRAINVAL_END]

    # align all sources to identical hourly timestamps
    common_index = (
        wave.index
        .intersection(mslp.index)
        .intersection(wind_u.index)
        .intersection(wind_v.index)
        .sort_values()
    )

    if common_index.empty:
        raise ValueError(
            "no timestamps are shared by the wave and atmospheric inputs"
        )

    wave = wave.loc[common_index]
    mslp = mslp.loc[common_index]
    wind_u = wind_u.loc[common_index]
    wind_v = wind_v.loc[common_index]

    check_finite_predictors("mslp", mslp)
    check_finite_predictors("wind-u", wind_u)
    check_finite_predictors("wind-v", wind_v)

    print(
        f"aligned {len(common_index):,} rows from "
        f"{common_index[0]} to {common_index[-1]}",
        flush=True,
    )

    outputs = []

    for window in sorted(set(args.windows)):
        outputs.append(
            create_window_file(
                mslp=mslp,
                wind_u=wind_u,
                wind_v=wind_v,
                wave=wave,
                window=window,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
        )

    print("=" * 72, flush=True)
    print(
        "train-validation tensor generation completed successfully",
        flush=True,
    )

    for output in outputs:
        print(f"created: {output}", flush=True)


if __name__ == "__main__":
    main()
