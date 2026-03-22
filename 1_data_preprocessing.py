# ---------------------------------------------------------------------------
# preprocess mslp and 10 m wind (u, v) for metascales work
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------

"""
preprocess mslp and 10 m wind (u, v) from coastdat / metascales netcdf files.

outputs:
    - stdanom_mslp.parquet
    - stdanom_windU.parquet
    - stdanom_windV.parquet

each file:
    - index: time (datetime64)
    - columns: "lat_lon" formatted as "{lat:.2f}_{lon:.2f}"
    - values: standardised anomalies (z-score per grid cell)
"""

from pathlib import Path

import xarray as xr
import pandas as pd
import numpy as np


# configuration

# wind data (u, v) – multi-year files
WIND_INPUT_DIR = Path("/work/gg0303/g260244/METAscales_data/wind_data")
WIND_FILE_PATTERN = "cD3.UV10M.META.*.nc"

# pmsl (mslp) data – already combined 1948–2023
MSLP_INPUT_DIR = Path("/work/gg0303/g260244/METAscales_data/st_data/0_25_res_10W20E40N65N")
MSLP_FILE_NAME = "combined_1948_2023.nc"

# output directory
OUTPUT_DIR = Path("/work/gg0303/g260244/metascales/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# cutoff date for handling variable-name change in wind data
CUTOFF_DATE = pd.to_datetime("1966-12-31 23:00:00")


# helper functions

def stack_to_dataframe(
    da: xr.DataArray,
    lat_name: str = "lat",
    lon_name: str = "lon"
) -> pd.DataFrame:
    """
    stack lat/lon into a single 'coord' dimension and convert to pandas.dataframe.

    result:
        index: time
        columns: "{lat:.2f}_{lon:.2f}"
    """
    # stack lat/lon into a single dimension
    stacked = da.stack(coord=(lat_name, lon_name))

    # convert to dataframe
    df = stacked.to_pandas()  # index: time, columns: multiindex(lat, lon)

    # rename columns from multiindex (lat, lon) to "lat_lon" string
    df.columns = [f"{lat:.2f}_{lon:.2f}" for lat, lon in df.columns]

    return df


def standardise_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    standardise each column: (x - mean) / std.
    nan-only columns are kept as nan.
    """
    mean = df.mean()
    std = df.std()

    # avoid division by zero: replace 0 with nan
    std_replaced = std.replace(0, np.nan)

    return (df - mean) / std_replaced


# wind: u and v at 10 m

def process_wind_data() -> None:
    """
    load multi-file wind dataset, handle variable-name switch, standardise, and save.
    """
    print("loading wind data...")

    file_pattern = str(WIND_INPUT_DIR / WIND_FILE_PATTERN)

    # open multi-file dataset
    ds = xr.open_mfdataset(
        file_pattern,
        combine="by_coords",
    )

    # handle name change around 1966-12-31 23:00:00
    print("handling u/v variable names before/after cutoff date...")
    windU = xr.where(
        ds["time"] <= CUTOFF_DATE,
        ds["U_10M"],
        ds["Ulat_10M"],
    )
    windV = xr.where(
        ds["time"] <= CUTOFF_DATE,
        ds["V_10M"],
        ds["Vlat_10M"],
    )

    # convert to dataframe (time × grid)
    print("converting wind u to dataframe...")
    df_windU = stack_to_dataframe(windU)

    print("converting wind v to dataframe...")
    df_windV = stack_to_dataframe(windV)

    # check for duplicate timestamps
    if df_windU.index.has_duplicates:
        print("wind u index has duplicates – keeping first occurrence.")
        df_windU = df_windU[~df_windU.index.duplicated(keep="first")]
    if df_windV.index.has_duplicates:
        print("wind v index has duplicates – keeping first occurrence.")
        df_windV = df_windV[~df_windV.index.duplicated(keep="first")]

    # standardise
    print("standardising wind u and v...")
    std_windU = standardise_by_column(df_windU)
    std_windV = standardise_by_column(df_windV)

    # sanity check for nans
    n_nans_u = std_windU.isna().sum().sum()
    n_nans_v = std_windV.isna().sum().sum()
    print(f"nan count in std_windU: {n_nans_u}")
    print(f"nan count in std_windV: {n_nans_v}")

    # save to parquet
    out_u = OUTPUT_DIR / "stdanom_windU_data.parquet"
    out_v = OUTPUT_DIR / "stdanom_windV_data.parquet"

    print(f"saving standardised wind u to: {out_u}")
    std_windU.to_parquet(out_u, compression="zstd")

    print(f"saving standardised wind v to: {out_v}")
    std_windV.to_parquet(out_v, compression="zstd")

    print("wind data processing finished.")


# mslp processing

def process_mslp_data() -> None:
    """
    load combined pmsl file, standardise anomalies, and save.
    """
    file_path = MSLP_INPUT_DIR / MSLP_FILE_NAME
    print(f"loading mslp data from: {file_path}")

    ds = xr.open_dataset(file_path)
    mslp = ds["PMSL"]

    # convert to dataframe (time × grid)
    print("converting mslp to dataframe...")
    df_mslp = stack_to_dataframe(mslp)

    # check for duplicate timestamps
    if df_mslp.index.has_duplicates:
        print("mslp index has duplicates – keeping first occurrence.")
        df_mslp = df_mslp[~df_mslp.index.duplicated(keep="first")]

    # standardise
    print("standardising mslp...")
    std_mslp = standardise_by_column(df_mslp)

    # sanity check
    n_nans = std_mslp.isna().sum().sum()
    print(f"nan count in std_mslp: {n_nans}")

    # save
    out_mslp = OUTPUT_DIR / "stdanom_mslp_data.parquet"
    print(f"saving standardised mslp to: {out_mslp}")
    std_mslp.to_parquet(out_mslp, compression="zstd")

    print("mslp data processing finished.")


# main

def main():
    print("===================================")
    print("  mslp + 10 m wind preprocessing")
    print("===================================")

    process_wind_data()
    process_mslp_data()

    print("all done.")


if __name__ == "__main__":
    main() MSLP + 10m wind preprocessing")
    print("===================================")

    process_wind_data()
    process_mslp_data()

    print("✅ All done.")


if __name__ == "__main__":
    main()