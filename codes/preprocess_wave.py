#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# preprocess mslp and 10 m wind (u, v) for metascales work
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March 2026
# ---------------------------------------------------------------------------

"""Preprocess wave data for the SEE and ELB deep-water locations.

expected input files
--------------------
../data/coastdat_wave_SEE.csv
../data/coastdat_wave_ELB.csv

output files
------------
../data/prepared_wave/wave_data_processed_SEE.parquet
../data/prepared_wave/wave_data_processed_ELB.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd


# only the two deep-water study locations are used in the revised analysis
POINTS_OF_INTEREST = ("SEE", "ELB")

# define input and output directories relative to the scripts directory
DATA_DIR = Path("../data")
OUTPUT_DIR = DATA_DIR / "prepared_wave"


def preprocess_wave_file(poi: str) -> Path:
    """Preprocess one wave file and return the saved parquet path."""
    input_path = DATA_DIR / f"coastdat_wave_{poi}.csv"
    output_path = OUTPUT_DIR / f"wave_data_processed_{poi}.parquet"

    print("=" * 72, flush=True)
    print(f"preprocessing wave data for {poi}", flush=True)
    print(f"input:  {input_path}", flush=True)
    print(f"output: {output_path}", flush=True)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"wave input file not found: {input_path}"
        )

    # load the wave time series and enforce a sorted datetime index
    wave_data = pd.read_csv(input_path, index_col=0)
    wave_data.index = pd.to_datetime(wave_data.index)
    wave_data.index.name = "time"
    wave_data.sort_index(inplace=True)

    if wave_data.index.has_duplicates:
        n_duplicates = int(wave_data.index.duplicated().sum())
        raise ValueError(
            f"{poi} contains {n_duplicates} duplicate timestamps"
        )

    print(
        f"loaded {len(wave_data):,} rows from "
        f"{wave_data.index[0]} to {wave_data.index[-1]}",
        flush=True,
    )

    # use the target name expected by the training workflow
    if "tm2" in wave_data.columns and "t0" not in wave_data.columns:
        wave_data.rename(columns={"tm2": "t0"}, inplace=True)

    # support older wave files where direction was stored as delta
    if "delta" in wave_data.columns and "thq" not in wave_data.columns:
        wave_data.rename(columns={"delta": "thq"}, inplace=True)

    required_columns = {"hs", "t0", "thq"}
    missing_columns = required_columns.difference(wave_data.columns)

    if missing_columns:
        raise KeyError(
            f"{poi} is missing required columns: {sorted(missing_columns)}"
        )

    # reproduce the original treatment of missing low-period values
    n_missing_t0 = int(wave_data["t0"].isna().sum())

    if n_missing_t0 > 0:
        minimum_t0 = wave_data["t0"].min(skipna=True)

        if not np.isfinite(minimum_t0):
            raise ValueError(
                f"{poi} contains no finite t0 values"
            )

        wave_data["t0"] = wave_data["t0"].fillna(minimum_t0)
        print(
            f"filled {n_missing_t0} missing t0 values with "
            f"the observed minimum ({minimum_t0})",
            flush=True,
        )

    # remove any remaining rows with missing target variables
    target_columns = ["hs", "t0", "thq"]
    missing_target_mask = wave_data[target_columns].isna().any(axis=1)
    n_missing_rows = int(missing_target_mask.sum())

    if n_missing_rows > 0:
        wave_data = wave_data.loc[~missing_target_mask].copy()
        print(
            f"removed {n_missing_rows} rows with remaining missing targets",
            flush=True,
        )
    else:
        print("no remaining missing wave targets found", flush=True)

    # convert direction to radians when the stored values appear to be degrees
    direction_max = wave_data["thq"].abs().max()

    if direction_max > 2 * np.pi + 1e-3:
        wave_data["thq"] = np.deg2rad(
            wave_data["thq"].to_numpy(dtype=np.float64)
        )
        print("converted thq from degrees to radians", flush=True)
    else:
        print("thq already appears to be in radians", flush=True)

    # encode circular direction as continuous sine and cosine targets
    wave_data["thq_sin"] = np.sin(wave_data["thq"])
    wave_data["thq_cos"] = np.cos(wave_data["thq"])

    # retain the variables used in the model workflow in a fixed order
    output_columns = ["hs", "t0", "thq", "thq_sin", "thq_cos"]
    wave_data = wave_data[output_columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wave_data.to_parquet(
        output_path,
        compression="zstd",
    )

    print(f"saved shape: {wave_data.shape}", flush=True)
    print(f"saved: {output_path}", flush=True)
    return output_path


def main() -> None:
    """Preprocess and save wave data for both deep-water locations."""
    print("starting wave preprocessing for see and elb", flush=True)

    saved_files = []

    for poi in POINTS_OF_INTEREST:
        saved_files.append(preprocess_wave_file(poi))

    print("=" * 72, flush=True)
    print("wave preprocessing completed successfully", flush=True)

    for saved_file in saved_files:
        print(f"saved file: {saved_file}", flush=True)


if __name__ == "__main__":
    main()