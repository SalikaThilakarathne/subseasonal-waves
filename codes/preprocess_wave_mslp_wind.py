# ---------------------------------------------------------------------------
# preprocess wave, mslp and 10 m wind (u, v) for metascales work
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

# wind data (u, v)
data_dir = Path("../data")
data_dir.mkdir(parents=True, exist_ok=True)
windU_file = "coastdat_windU_1948_2023.nc"
windV_file = "coastdat_windV_1948_2023.nc"

# pmsl (mslp) data
mslp_file = "coastdat_mslp_1948_2023.nc"

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

# wind: u and v at 10 m--have we checked?...


def process_wind_data() -> None:
    """
    Load preprocessed windU and windV NetCDF files, convert to dataframes, standardise, and save as parquet.
    """
    print("loading preprocessed windU / windV data...")

    # open the U and V wind files
    dsU = xr.open_dataset(data_dir / windU_file)
    dsV = xr.open_dataset(data_dir / windV_file)

    # extract the DataArray from each dataset
    # If you know the variable names, use them explicitly, e.g. dsU["windU"]
    # Otherwise, grab the first (and presumably only) data variable:
    var_name_u = list(dsU.data_vars)[0]
    var_name_v = list(dsV.data_vars)[0]

    windU = dsU[var_name_u]
    windV = dsV[var_name_v]

    print(f"using variable '{var_name_u}' from {windU_file}")
    print(f"using variable '{var_name_v}' from {windV_file}")

    # convert to dataframe (time × grid)---you get a dfs
    print("converting wind U to dataframe...")
    df_windU = stack_to_dataframe(windU, lat_name="lat", lon_name="lon")

    print("converting wind V to dataframe...")
    df_windV = stack_to_dataframe(windV, lat_name="lat", lon_name="lon")

    # check for duplicate timestamps
    if df_windU.index.has_duplicates:
        print("wind U index has duplicates – keeping first occurrence.")
        df_windU = df_windU[~df_windU.index.duplicated(keep="first")]

    if df_windV.index.has_duplicates:
        print("wind V index has duplicates – keeping first occurrence.")
        df_windV = df_windV[~df_windV.index.duplicated(keep="first")]

    # standardise
    print("standardising wind U and V...")
    std_windU = standardise_by_column(df_windU)
    std_windV = standardise_by_column(df_windV)

    # check for NaNs
    n_nans_u = std_windU.isna().sum().sum()
    n_nans_v = std_windV.isna().sum().sum()
    print(f"NaN count in std_windU: {n_nans_u}")
    print(f"NaN count in std_windV: {n_nans_v}")

    # save to parquet--better to work with the ml models...
    out_u = data_dir / "stdanom_windU_data.parquet"
    out_v = data_dir / "stdanom_windV_data.parquet"

    print(f"saving standardised wind U to: {out_u}")
    std_windU.to_parquet(out_u, compression="zstd")

    print(f"saving standardised wind V to: {out_v}")
    std_windV.to_parquet(out_v, compression="zstd")

    print("wind data processing finished.")


# mslp processing

def process_mslp_data() -> None:
    """
    load combined pmsl file, standardise anomalies, and save.
    """
    file_path = data_dir / mslp_file
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
    out_mslp = data_dir / "stdanom_mslp_data.parquet"
    print(f"saving standardised mslp to: {out_mslp}")
    std_mslp.to_parquet(out_mslp, compression="zstd")

    print("mslp data processing finished.")

def process_wave_data():
    """
    Load and preprocess wave data (hs, tm2, thq) for all locations.
    Converts selected variables into a clean time‑indexed parquet file.
    """

    # match files like:  xx.SEE.nc-- here SEE is the location. so the last 3 char should represents the location...
    files = sorted(glob.glob(os.path.join(data_dir, wave_file_pattern)))

    if not files:
        print("No wave files found. Check directory or pattern!")
        return

    for f in files:
        print(f"Processing: {f}")

        ds = xr.open_dataset(f)

        # Select hs, tm2, thq → convert to DataFrame
        df = ds[wave_variables].to_dataframe().reset_index()

        # Index by time and keep only the selected variables
        df = df.set_index("time")[wave_variables]

        # Extract location code (e.g. SEE, BSH, HELG, etc.)
        location_code = os.path.basename(f).split(".")[-2]

        # Output path
        mslp_file = f"coastdat_wave_{location_code}.parquet"
        output_file = data_dir / mslp_file

        # Save as parquet (correct!)
        df.to_parquet(output_file, compression="zstd")
        print(f"Saved → {output_file}")


# main---to run the entire workflow...

def main():
    print("===================================")
    print("mslp + 10 m  wind preprocessing")
    print("===================================")

    process_wind_data()
    process_mslp_data()
    # process_wave_data()

    print("all done!!!")

if __name__ == "__main__":
    main()