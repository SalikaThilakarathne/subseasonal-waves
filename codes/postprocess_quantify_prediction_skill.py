#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# quantify prediction skill using paired block bootstrap confidence intervals
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# October 2026
# ---------------------------------------------------------------------------


"""
Quantify skill for all atmosphere-wave prediction parquet files.

The script scans a results directory, identifies transformer and mlp prediction files, calculates skill for hs, t0, and wave direction, and estimates paired block-bootstrap confidence intervals. observed and predicted values are always resampled together so their temporal correspondence is preserved.

Expected prediction columns
---------------------------
hs_obs, hs_pred
t0_obs, t0_pred
thq_deg_obs, thq_deg_pred

Example
-------
python -u quantify_prediction_skill.py --results_dir ../results --output_dir ../results/skill_metrics --n_boot 10000 --block_hours 168 --seed 42
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


file_pattern = re.compile(
    r"^(?P<date>\d{4}_\d{2}_\d{2})_"
    r"(?P<model>"
    r"atmoswaveT|"
    r"atmoswave-transformer|"
    r"atmoswave_bMLP|"
    r"atmoswaveCNNTransformer"
    r")_predictions_"
    r"(?P<poi>SEE|ELB)_"
    r"w(?P<window>\d{2})_"
    r"sd(?P<seed>\d{4})\.parquet$"
)


variables = {
    "hs": ("hs_obs", "hs_pred", "linear"),
    "t0": ("t0_obs", "t0_pred", "linear"),
    "thq": ("thq_deg_obs", "thq_deg_pred", "circular"),
}


def parse_arguments() -> argparse.Namespace:
    """Read input, output, and bootstrap settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("../results"),
        help="directory containing prediction parquet files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../results/skill_metrics"),
        help="directory used for metric summaries",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_predictions_*.parquet",
        help="glob pattern used to find prediction files",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=10_000,
        help="number of bootstrap replicates",
    )
    parser.add_argument(
        "--block_hours",
        type=int,
        default=24 * 7,
        help="length of each paired temporal block in hours",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used for bootstrap resampling",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing metric output files",
    )
    return parser.parse_args()


def parse_filename(path: Path) -> dict:
    """Extract experiment metadata from one prediction filename."""
    match = file_pattern.match(path.name)

    if match is None:
        raise ValueError(
            f"prediction filename does not follow the expected convention: {path.name}"
        )

    metadata = match.groupdict()
    model_name = metadata["model"]

    if model_name in {
        "atmoswaveT",
        "atmoswave-transformer",
    }:
        metadata["model"] = "transformer"
    
    elif model_name == "atmoswave_bMLP":
        metadata["model"] = "mlp"
    
    elif model_name == "atmoswaveCNNTransformer":
        metadata["model"] = "cnn_transformer"


    metadata["window_hours"] = int(metadata.pop("window"))
    metadata["seed"] = int(metadata["seed"])
    return metadata


def signed_circular_error_deg(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Return signed shortest-angle error in the interval [-180, 180)."""
    return (predicted - observed + 180.0) % 360.0 - 180.0


def linear_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate conventional metrics for a linear variable."""
    correlation = spearmanr(observed, predicted).statistic

    return {
        "rmse": root_mean_squared_error(observed, predicted),
        "mae": mean_absolute_error(observed, predicted),
        "bias": float(np.mean(predicted - observed)),
        "r2": r2_score(observed, predicted),
        "spearman": float(correlation),
    }


def circular_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate angular error metrics for direction in degrees."""
    errors = signed_circular_error_deg(observed, predicted)

    # r2 and ordinary rank correlation are not used for wrapped angular values
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "r2": np.nan,
        "spearman": np.nan,
    }


def calculate_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    variable_type: str,
) -> dict[str, float]:
    """Dispatch to linear or circular metric calculations."""
    if variable_type == "linear":
        return linear_metrics(observed, predicted)

    if variable_type == "circular":
        return circular_metrics(observed, predicted)

    raise ValueError(f"unsupported variable type: {variable_type}")


def prepare_complete_blocks(
    dataframe: pd.DataFrame,
    observed_column: str,
    predicted_column: str,
    block_hours: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create complete paired blocks without joining across time gaps."""
    selected = dataframe[[observed_column, predicted_column]].copy()
    selected = selected.replace([np.inf, -np.inf], np.nan).dropna()

    if selected.empty:
        raise ValueError(
            f"no finite paired data for {observed_column} and {predicted_column}"
        )

    if not isinstance(selected.index, pd.DatetimeIndex):
        selected.index = pd.to_datetime(selected.index)

    selected.sort_index(inplace=True)

    # assign a new segment whenever timestamps are not exactly one hour apart
    breaks = selected.index.to_series().diff().ne(pd.Timedelta(hours=1))
    breaks.iloc[0] = True
    segment_ids = breaks.cumsum().to_numpy()

    observed_blocks = []
    predicted_blocks = []

    for segment_id in np.unique(segment_ids):
        segment = selected.iloc[segment_ids == segment_id]
        n_blocks = len(segment) // block_hours

        if n_blocks == 0:
            continue

        n_used = n_blocks * block_hours
        observed_values = segment[observed_column].to_numpy()[:n_used]
        predicted_values = segment[predicted_column].to_numpy()[:n_used]

        observed_blocks.append(
            observed_values.reshape(n_blocks, block_hours)
        )
        predicted_blocks.append(
            predicted_values.reshape(n_blocks, block_hours)
        )

    if not observed_blocks:
        raise ValueError(
            f"no complete {block_hours}-hour blocks were found"
        )

    observed_blocks = np.concatenate(observed_blocks, axis=0)
    predicted_blocks = np.concatenate(predicted_blocks, axis=0)
    n_used = int(observed_blocks.size)

    return observed_blocks, predicted_blocks, n_used


def paired_block_bootstrap(
    observed_blocks: np.ndarray,
    predicted_blocks: np.ndarray,
    variable_type: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate metric distributions by resampling paired blocks."""
    n_blocks = observed_blocks.shape[0]
    distributions = {
        "rmse": np.empty(n_boot, dtype=np.float64),
        "mae": np.empty(n_boot, dtype=np.float64),
        "bias": np.empty(n_boot, dtype=np.float64),
        "r2": np.full(n_boot, np.nan, dtype=np.float64),
        "spearman": np.full(n_boot, np.nan, dtype=np.float64),
    }

    for replicate in range(n_boot):
        block_indices = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )
        observed = observed_blocks[block_indices].reshape(-1)
        predicted = predicted_blocks[block_indices].reshape(-1)
        metrics = calculate_metrics(
            observed,
            predicted,
            variable_type,
        )

        for metric_name, metric_value in metrics.items():
            distributions[metric_name][replicate] = metric_value

    return distributions


def summarise_metric(
    estimate: float,
    bootstrap_values: np.ndarray,
) -> dict[str, float]:
    """Summarise one original estimate and its bootstrap distribution."""
    finite_values = bootstrap_values[np.isfinite(bootstrap_values)]

    if finite_values.size == 0:
        return {
            "estimate": estimate,
            "bootstrap_mean": np.nan,
            "bootstrap_std": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "bootstrap_median": np.nan,
        }

    lower, upper = np.percentile(
        finite_values,
        [2.5, 97.5],
    )

    return {
        "estimate": estimate,
        "bootstrap_mean": float(np.mean(finite_values)),
        "bootstrap_std": float(np.std(finite_values, ddof=1)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "bootstrap_median": float(np.median(finite_values)),
    }


def analyse_prediction_file(
    prediction_path: Path,
    n_boot: int,
    block_hours: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Calculate skill and uncertainty for one prediction file."""
    metadata = parse_filename(prediction_path)
    predictions = pd.read_parquet(prediction_path)

    if not isinstance(predictions.index, pd.DatetimeIndex):
        predictions.index = pd.to_datetime(predictions.index)

    rows = []

    for variable, (
        observed_column,
        predicted_column,
        variable_type,
    ) in variables.items():
        missing_columns = {
            observed_column,
            predicted_column,
        }.difference(predictions.columns)

        if missing_columns:
            print(
                f"skipping {variable} in {prediction_path.name}; "
                f"missing {sorted(missing_columns)}",
                flush=True,
            )
            continue

        observed_blocks, predicted_blocks, n_used = prepare_complete_blocks(
            predictions,
            observed_column,
            predicted_column,
            block_hours,
        )
        observed = observed_blocks.reshape(-1)
        predicted = predicted_blocks.reshape(-1)
        original_metrics = calculate_metrics(
            observed,
            predicted,
            variable_type,
        )

        # derive an independent but reproducible stream for each experiment variable
        variable_seed = (
            bootstrap_seed
            + metadata["seed"] * 10_000
            + metadata["window_hours"] * 100
            + sum(ord(character) for character in variable)
        )
        rng = np.random.default_rng(variable_seed)
        distributions = paired_block_bootstrap(
            observed_blocks,
            predicted_blocks,
            variable_type,
            n_boot,
            rng,
        )

        for metric_name, estimate in original_metrics.items():
            summary = summarise_metric(
                estimate,
                distributions[metric_name],
            )
            rows.append(
                {
                    "source_file": prediction_path.name,
                    **metadata,
                    "variable": variable,
                    "variable_type": variable_type,
                    "metric": metric_name,
                    **summary,
                    "n_observations": n_used,
                    "n_blocks": observed_blocks.shape[0],
                    "block_hours": block_hours,
                    "n_bootstrap": n_boot,
                    "bootstrap_seed": bootstrap_seed,
                }
            )

    return pd.DataFrame(rows)


def create_wide_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create one compact row per experiment, variable, and metric."""
    index_columns = [
        "date",
        "model",
        "poi",
        "window_hours",
        "seed",
        "variable",
        "metric",
    ]
    value_columns = [
        "ci95_lower",          # bootstrap 2.5th percentile
        "bootstrap_median",    # bootstrap 50th percentile
        "ci95_upper",          # bootstrap 97.5th percentile
        "bootstrap_mean",
        "bootstrap_std",
        "n_observations",
        "n_blocks",
        "block_hours",
    ]

    return metrics[index_columns + value_columns].sort_values(
        index_columns
    )


def main() -> None:
    """Analyse every matching prediction file and save reusable summaries."""
    args = parse_arguments()

    if args.n_boot < 1:
        raise ValueError("--n_boot must be at least 1")

    if args.block_hours < 1:
        raise ValueError("--block_hours must be at least 1")

    prediction_files = sorted(
        args.results_dir.glob(args.pattern)
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"no files match {args.results_dir / args.pattern}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    detailed_path = args.output_dir / "prediction_skill_block_bootstrap.parquet"
    csv_path = args.output_dir / "prediction_skill_block_bootstrap.csv"

    for output_path in (detailed_path, csv_path):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} exists; use --overwrite to replace it"
            )

    print("=" * 72, flush=True)
    print("prediction skill quantification", flush=True)
    print("=" * 72, flush=True)
    print(f"prediction files: {len(prediction_files)}", flush=True)
    print(f"bootstrap replicates: {args.n_boot:,}", flush=True)
    print(f"block length: {args.block_hours} hours", flush=True)
    print(f"bootstrap seed: {args.seed}", flush=True)

    started = time.perf_counter()
    results = []

    for file_number, prediction_path in enumerate(
        prediction_files,
        start=1,
    ):
        print(
            f"[{file_number}/{len(prediction_files)}] "
            f"analysing {prediction_path.name}",
            flush=True,
        )

        try:
            file_metrics = analyse_prediction_file(
                prediction_path,
                args.n_boot,
                args.block_hours,
                args.seed,
            )
            results.append(file_metrics)
        except ValueError as error:
            print(
                f"skipping {prediction_path.name}: {error}",
                flush=True,
            )

    if not results:
        raise RuntimeError("no prediction files could be analysed")

    metrics = pd.concat(
        results,
        ignore_index=True,
    )
    summary = create_wide_summary(metrics)

    summary.to_parquet(
        detailed_path,
        compression="zstd",
        index=False,
    )
    summary.to_csv(
        csv_path,
        index=False,
    )

    elapsed_minutes = (
        time.perf_counter() - started
    ) / 60

    print(f"saved: {detailed_path}", flush=True)
    print(f"saved: {csv_path}", flush=True)
    print(f"rows saved: {len(summary):,}", flush=True)
    print(f"elapsed time: {elapsed_minutes:.2f} min", flush=True)
    print("skill quantification completed successfully", flush=True)


if __name__ == "__main__":
    main()
