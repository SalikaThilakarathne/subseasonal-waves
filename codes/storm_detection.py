#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# wave storm identification: https://doi.org/10.1017/cft.2025.10012
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March 2026
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utils

from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from scipy.stats import spearmanr


def detect_wave_storms(
    wave_data: pd.DataFrame,
    freq_hours: float = 1,
    threshold1: float = 1.5,
    threshold2: float = 2.5,
    min_duration: float = 6,
    min_gap: float = 24
) -> pd.DataFrame:
    """
    Detects coastal storms based on wave height time series using two thresholds,
    minimum storm duration, and meteorological independence criteria.

    Parameters:
    - wave_data (pd.DataFrame): DataFrame indexed by datetime with at least 'hs' column.
                                Optional: 't0', 'thq' columns for storm summary.
    - freq_hours (int): Sampling interval of the wave data in hours.
    - threshold1 (float): Minimum Hs to consider part of a storm (for duration).
    - threshold2 (float): Minimum max Hs to count as a valid storm (impactful event).
    - min_duration (int): Minimum storm duration in hours.
    - min_gap (int): Minimum separation between storms to consider them independent.

    Returns:
    - pd.DataFrame: Storm event summary including:
        'storm_id', 'start_time', 'end_time', 'duration_hours', 'max_hs', 'mean_hs',
        'mean_t0', 'mean_thq'
    """
    wave_data = wave_data.copy()
    wave_data = wave_data.sort_index()

    storm_events = []
    start_time = None

    for current_time, row in wave_data.iterrows():
        if row['hs'] > threshold1:
            if start_time is None:
                start_time = current_time
        else:
            if start_time is not None:
                duration = (current_time - start_time).total_seconds() / 3600
                if duration >= min_duration:
                    event = wave_data.loc[start_time:current_time - pd.Timedelta(hours=freq_hours)]
                    storm_events.append(event)
                start_time = None

    if start_time is not None:
        duration = (wave_data.index[-1] - start_time).total_seconds() / 3600
        if duration >= min_duration:
            event = wave_data.loc[start_time:]
            storm_events.append(event)

    event_summary = []

    for event in storm_events:
        max_hs = event['hs'].max()
        time_max_hs = event['hs'].idxmax()
        if max_hs >= threshold2:
            start_time = event.index[0]
            end_time = event.index[-1]
            duration = (end_time - start_time).total_seconds() / 3600
            mean_hs = event['hs'].mean()
            mean_t0 = event['t0'].mean() if 't0' in event.columns else None
            mean_thq = event['thq'].mean() if 'thq' in event.columns else None

            event_summary.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration,
                'max_hs': max_hs,
                'time_max_hs': time_max_hs,
                'mean_hs': mean_hs,
                'mean_t0': mean_t0,
                'mean_thq': mean_thq
            })

    merged_events = []
    if event_summary:
        current = event_summary[0]
        for next_event in event_summary[1:]:
            gap = (next_event['start_time'] - current['end_time']).total_seconds() / 3600
            if gap <= min_gap:
                current['end_time'] = next_event['end_time']
                current['duration_hours'] = (current['end_time'] - current['start_time']).total_seconds() / 3600
                if next_event['max_hs'] > current['max_hs']:
                    current['max_hs'] = next_event['max_hs']
                    current['time_max_hs'] = next_event['time_max_hs']
                current['mean_hs'] = (current['mean_hs'] + next_event['mean_hs']) / 2
                if current['mean_t0'] is not None and next_event['mean_t0'] is not None:
                    current['mean_t0'] = (current['mean_t0'] + next_event['mean_t0']) / 2
                if current['mean_thq'] is not None and next_event['mean_thq'] is not None:
                    current['mean_thq'] = (current['mean_thq'] + next_event['mean_thq']) / 2
            else:
                merged_events.append(current)
                current = next_event
        merged_events.append(current)

    for i, event in enumerate(merged_events, start=1):
        event['storm_id'] = i

    return pd.DataFrame(merged_events)

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def simple_windrose(ax, directions, n_dirs=16, facecolor='C0', edgecolor='k'):
    """
    Plot a simple frequency windrose on a given polar Axes.
    """
    directions = np.asarray(directions)

    # Ensure angles are in [0, 2π)
    directions = np.mod(directions, 2 * np.pi)

    # Direction bins
    bin_edges = np.linspace(0, 2 * np.pi, n_dirs + 1)
    hist, _ = np.histogram(directions, bins=bin_edges)

    # Convert to frequencies (optional, just normalization)
    freq = hist / hist.sum() if hist.sum() > 0 else hist

    # Bin centers
    theta = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    width = (2 * np.pi) / n_dirs

    # Plot as bars
    bars = ax.bar(theta, freq, width=width,
                  align='center',
                  facecolor=facecolor,
                  edgecolor=edgecolor,
                  linewidth=0.3)

    # Simple styling
    ax.set_yticklabels([])  # hide radial labels
    ax.set_xticklabels([])  # hide angle labels (for tiny insets)
    ax.set_ylim(0, freq.max() * 1.1 if freq.max() > 0 else 1)

def plot_storm(data_obs, data_pred, date1, date2, year, metrics_on=False, ax=None, figsize=(4.2, 2.2), dpi=600, fig_num=""):
    """Plot storm with Hs (left y-axis) and T0 (right y-axis)."""

    # ---- define centered 21-day window ----
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    midpoint = date1 + (date2 - date1) / 2
    
    window_start = midpoint - pd.Timedelta(days=10.5)
    window_end   = midpoint + pd.Timedelta(days=10.5)
    
    # extract data for fixed 21‑day window
    hs_obs  = data_obs.loc[window_start:window_end, "hs"]
    hs_pred = data_pred.loc[window_start:window_end, "hs"]
    t0_obs  = data_obs.loc[window_start:window_end, "t0"]
    t0_pred = data_pred.loc[window_start:window_end, "t0"]
    
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig, ax1 = None, ax

    # ---- rectangle highlighting storm period ----
    ax1.axvspan(date1, date2, color=utils.hereon_cp[1], alpha=0.05, linewidth=0)

    # ---- Left axis: Hs ----
    ax1.plot(hs_obs,  lw=0.4, color=utils.hereon_cp[2], linestyle='--', label='H$_s$ hcst')
    ax1.plot(hs_pred, lw=0.4, color=utils.hereon_cp[2], linestyle='-',  label='H$_s$ pred')

    ax1.spines['top'].set_visible(False)
    ax1.yaxis.tick_left()
    ax1.xaxis.tick_bottom()
    ax1.minorticks_off()
    ax1.set_ylim(0, 10)
    ax1.set_yticks([2, 4, 6, 8])

    ax1.set_ylabel('H$_s$ [m]') #Significant wave height, 
    ax1.yaxis.grid(True, linestyle='-', linewidth=0.05)
    
    ax1.spines['left'].set_color(utils.hereon_cp[2])
    ax1.tick_params(axis='y', colors=utils.hereon_cp[2])
    ax1.yaxis.label.set_color(utils.hereon_cp[2])

    ax1.text(0.05, 0.95, fig_num, transform=ax1.transAxes,
         ha='left', va='top', color='grey') #, fontsize=5

    ax1.text(0.5, 0.95, str(year),transform=ax1.transAxes,
             ha='center', va='top',color='black', fontweight='bold')  # year of the event

    # ---- Right axis: T0 ----
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.minorticks_off()

    ax2.plot(t0_obs,  lw=0.4, color=utils.hereon_cp[5], linestyle='--',  label='T$_{m02}$ hcst')
    ax2.plot(t0_pred, lw=0.4, color=utils.hereon_cp[5], linestyle='-', label='T$_{m02}$ pred')

    ax2.set_ylim(0, 15) #0.1
    ax2.set_yticks([3, 6, 9, 12])
    ax2.set_ylabel('T$_{m02}$ [s]') #Mean wave period, 

    # ax2.spines['right'].set_color(utils.hereon_cp[5])
    ax2.tick_params(axis='y', colors=utils.hereon_cp[5])
    ax2.yaxis.label.set_color(utils.hereon_cp[5])

    # ---- X-axis formatting ----
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    if metrics_on:
        # hs-metrics
        mask = (hs_obs.index >= date1) & (hs_obs.index <= date2)
        metrics_hs = (
            # f"$\\bf{{Hs}}$\n"
            f"RMSE: {root_mean_squared_error(hs_obs[mask], hs_pred[mask]):.2f} m\n"
            f"Bias: {np.mean(hs_pred[mask] - hs_obs[mask]):.2f} m\n"
            f"MAE: {mean_absolute_error(hs_obs[mask], hs_pred[mask]):.2f} m\n"
            f"R²: {r2_score(hs_obs[mask], hs_pred[mask]):.2f}\n"
            f"rₛ: {spearmanr(hs_obs[mask], hs_pred[mask])[0]:.2f}\n"
            f"N: {len(hs_obs[mask])}"
        )
        
        at_hs = AnchoredText(
            metrics_hs,
            loc='lower left',
            prop=dict(size=5, color=utils.hereon_cp[2]),
            frameon=False
        )
        ax1.add_artist(at_hs)
    
        #t0 matrix
        metrics_t0 = (
            # f"Tₘ₀₂\n"
            f"RMSE: {root_mean_squared_error(t0_obs[mask], t0_pred[mask]):.2f} s\n"
            f"Bias: {np.mean(t0_pred[mask] - t0_obs[mask]):.2f} s\n"
            f"MAE: {mean_absolute_error(t0_obs[mask], t0_pred[mask]):.2f} s\n"
            f"R²: {r2_score(t0_obs[mask], t0_pred[mask]):.2f}\n"
            f"rₛ: {spearmanr(t0_obs[mask], t0_pred[mask])[0]:.2f}\n"
            f"N: {len(t0_obs[mask])}"
        )
        
        at_t0 = AnchoredText(
            metrics_t0,
            loc='lower right',
            prop=dict(size=5, color=utils.hereon_cp[5]),
            frameon=False
        )
        ax2.add_artist(at_t0)

    ticks = ax1.get_xticks()
    labels = [mdates.num2date(t).strftime('%b %d') for t in ticks]
    if labels:
        labels[-1] = mdates.num2date(ticks[-1]).strftime('%b %d, %Y')
        ax1.set_xticklabels(labels)

    # ---- Combined legend ----
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=6, ncol=2, frameon=False)

    if fig is not None:
        fig.tight_layout()
    return fig, ax1, ax2


def plot_storm_ensemble(
    data_obs,
    data_pred,
    date1,
    date2,
    year,
    metrics_on=False,
    ax=None,
    figsize=(4.2, 2.2),
    dpi=600,
    fig_num="",
):
    """
    Plot observed and ensemble-predicted Hs and Tm02 for a storm event.

    The ensemble mean is shown as a solid line. The minimum-to-maximum
    range across model seeds is shown as a shaded band.

    Parameters
    ----------
    data_obs : pandas.DataFrame
        Observed data with columns:
        - hs
        - t0

    data_pred : pandas.DataFrame
        Prepared ensemble prediction with columns:
        - hs
        - hs_min
        - hs_max
        - t0
        - t0_min
        - t0_max

    date1, date2 : str or pandas.Timestamp
        Start and end timestamps of the highlighted storm period.

    year : str or int
        Event year shown at the top of the panel.

    metrics_on : bool, default=False
        If True, calculate metrics over the highlighted storm period.
        Metrics are calculated using the ensemble-mean predictions.

    ax : matplotlib.axes.Axes, optional
        Existing axis used for a multi-panel figure.

    figsize : tuple, default=(4.2, 2.2)
        Figure size when a new figure is created.

    dpi : int, default=600
        Figure resolution when a new figure is created.

    fig_num : str, default=""
        Panel label, for example "(a)".

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object. Returns None when an existing axis is supplied.

    ax1 : matplotlib.axes.Axes
        Left axis showing significant wave height.

    ax2 : matplotlib.axes.Axes
        Right axis showing mean wave period.
    """
    # validate the inputs
    if data_obs is None or data_obs.empty:
        raise ValueError(
            "data_obs must be a non-empty DataFrame"
        )

    if data_pred is None or data_pred.empty:
        raise ValueError(
            "data_pred must be a non-empty DataFrame"
        )

    required_obs_columns = {
        "hs",
        "t0",
    }

    required_pred_columns = {
        "hs",
        "hs_min",
        "hs_max",
        "t0",
        "t0_min",
        "t0_max",
    }

    missing_obs = required_obs_columns.difference(
        data_obs.columns
    )

    missing_pred = required_pred_columns.difference(
        data_pred.columns
    )

    if missing_obs:
        raise KeyError(
            "data_obs is missing columns: "
            f"{sorted(missing_obs)}"
        )

    if missing_pred:
        raise KeyError(
            "data_pred is missing columns: "
            f"{sorted(missing_pred)}"
        )

    # copy the inputs so that their original indices are not modified
    data_obs = data_obs.copy()
    data_pred = data_pred.copy()

    if not isinstance(
        data_obs.index,
        pd.DatetimeIndex,
    ):
        data_obs.index = pd.to_datetime(
            data_obs.index
        )

    if not isinstance(
        data_pred.index,
        pd.DatetimeIndex,
    ):
        data_pred.index = pd.to_datetime(
            data_pred.index
        )

    data_obs.sort_index(
        inplace=True
    )

    data_pred.sort_index(
        inplace=True
    )

    # define the highlighted event and centred 21-day plotting window
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)

    if date2 < date1:
        raise ValueError(
            "date2 must be later than or equal to date1"
        )

    midpoint = date1 + (date2 - date1) / 2

    window_start = midpoint - pd.Timedelta(
        days=10.5
    )

    window_end = midpoint + pd.Timedelta(
        days=10.5
    )

    # retain timestamps shared by observations and predictions
    common_index = data_obs.index.intersection(
        data_pred.index
    ).sort_values()

    common_index = common_index[
        (common_index >= window_start)
        & (common_index <= window_end)
    ]

    if common_index.empty:
        raise ValueError(
            "no common timestamps are available "
            "within the selected 21-day window"
        )

    # extract observations
    hs_obs = data_obs.loc[
        common_index,
        "hs",
    ].astype(float)

    t0_obs = data_obs.loc[
        common_index,
        "t0",
    ].astype(float)

    # extract ensemble means and seed ranges
    hs_mean = data_pred.loc[
        common_index,
        "hs",
    ].astype(float)

    hs_min = data_pred.loc[
        common_index,
        "hs_min",
    ].astype(float)

    hs_max = data_pred.loc[
        common_index,
        "hs_max",
    ].astype(float)

    t0_mean = data_pred.loc[
        common_index,
        "t0",
    ].astype(float)

    t0_min = data_pred.loc[
        common_index,
        "t0_min",
    ].astype(float)

    t0_max = data_pred.loc[
        common_index,
        "t0_max",
    ].astype(float)

    # ensure the lower and upper bounds are correctly ordered
    hs_lower = pd.concat(
        [hs_min, hs_max],
        axis=1,
    ).min(axis=1)

    hs_upper = pd.concat(
        [hs_min, hs_max],
        axis=1,
    ).max(axis=1)

    t0_lower = pd.concat(
        [t0_min, t0_max],
        axis=1,
    ).min(axis=1)

    t0_upper = pd.concat(
        [t0_min, t0_max],
        axis=1,
    ).max(axis=1)

    # create a new figure or use the provided axis
    if ax is None:
        fig, ax1 = plt.subplots(
            figsize=figsize,
            dpi=dpi,
        )
    else:
        fig = None
        ax1 = ax

    # highlight the storm period
    ax1.axvspan(
        date1,
        date2,
        color=utils.hereon_cp[1],
        alpha=0.05,
        linewidth=0,
        zorder=0,
    )

    # ------------------------------------------------------------------
    # left axis: significant wave height
    # ------------------------------------------------------------------
    ax1.fill_between(
        common_index,
        hs_lower.to_numpy(),
        hs_upper.to_numpy(),
        color=utils.hereon_cp[2],
        alpha=0.20,
        linewidth=0,
        # label="H$_s$ seed range",
        zorder=1,
    )

    ax1.plot(
        common_index,
        hs_obs.to_numpy(),
        lw=0.4,
        color=utils.hereon_cp[2],
        linestyle="--",
        label="H$_s$ hcst",
        zorder=3,
    )

    ax1.plot(
        common_index,
        hs_mean.to_numpy(),
        lw=0.4,
        color=utils.hereon_cp[2],
        linestyle="-",
        label="H$_s$ mean",
        zorder=4,
    )

    ax1.spines["top"].set_visible(
        False
    )

    ax1.yaxis.tick_left()
    ax1.xaxis.tick_bottom()
    ax1.minorticks_off()

    ax1.set_xlim(
        window_start,
        window_end,
    )

    ax1.set_ylim(
        0,
        10,
    )

    ax1.set_yticks(
        [2, 4, 6, 8]
    )

    ax1.set_ylabel(
        "H$_s$ [m]"
    )

    ax1.yaxis.grid(
        True,
        linestyle="-",
        linewidth=0.05,
    )

    ax1.spines["left"].set_color(
        utils.hereon_cp[2]
    )

    ax1.tick_params(
        axis="y",
        colors=utils.hereon_cp[2],
    )

    ax1.yaxis.label.set_color(
        utils.hereon_cp[2]
    )

    # panel label
    ax1.text(
        0.05,
        0.95,
        fig_num,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        color="grey",
    )

    # event year
    ax1.text(
        0.5,
        0.95,
        str(year),
        transform=ax1.transAxes,
        ha="center",
        va="top",
        color="black",
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # right axis: mean wave period
    # ------------------------------------------------------------------
    ax2 = ax1.twinx()

    ax2.fill_between(
        common_index,
        t0_lower.to_numpy(),
        t0_upper.to_numpy(),
        color=utils.hereon_cp[5],
        alpha=0.20,
        linewidth=0,
        # label="T$_{m02}$ seed range",
        zorder=1,
    )

    ax2.plot(
        common_index,
        t0_obs.to_numpy(),
        lw=0.4,
        color=utils.hereon_cp[5],
        linestyle="--",
        label="T$_{m02}$ hcst",
        zorder=3,
    )

    ax2.plot(
        common_index,
        t0_mean.to_numpy(),
        lw=0.4,
        color=utils.hereon_cp[5],
        linestyle="-",
        label="T$_{m02}$ mean",
        zorder=4,
    )

    ax2.spines["top"].set_visible(
        False
    )

    ax2.yaxis.tick_right()
    ax2.minorticks_off()

    ax2.set_ylim(
        0,
        15,
    )

    ax2.set_yticks(
        [3, 6, 9, 12]
    )

    ax2.set_ylabel(
        "T$_{m02}$ [s]"
    )

    ax2.tick_params(
        axis="y",
        colors=utils.hereon_cp[5],
    )

    ax2.yaxis.label.set_color(
        utils.hereon_cp[5]
    )

    # ------------------------------------------------------------------
    # storm-period metrics calculated from the ensemble mean
    # ------------------------------------------------------------------
    if metrics_on:
        event_index = common_index[
            (common_index >= date1)
            & (common_index <= date2)
        ]

        if event_index.empty:
            raise ValueError(
                "the highlighted storm period contains "
                "no common observation-prediction timestamps"
            )

        hs_event_obs = hs_obs.loc[
            event_index
        ]

        hs_event_pred = hs_mean.loc[
            event_index
        ]

        t0_event_obs = t0_obs.loc[
            event_index
        ]

        t0_event_pred = t0_mean.loc[
            event_index
        ]

        # remove missing values separately for each variable
        hs_valid = (
            hs_event_obs.notna()
            & hs_event_pred.notna()
        )

        t0_valid = (
            t0_event_obs.notna()
            & t0_event_pred.notna()
        )

        n_hs = int(
            hs_valid.sum()
        )

        n_t0 = int(
            t0_valid.sum()
        )

        if n_hs < 2:
            raise ValueError(
                "fewer than two valid Hs values are available "
                "for the storm-period metrics"
            )

        if n_t0 < 2:
            raise ValueError(
                "fewer than two valid Tm02 values are available "
                "for the storm-period metrics"
            )

        hs_obs_valid = hs_event_obs[
            hs_valid
        ]

        hs_pred_valid = hs_event_pred[
            hs_valid
        ]

        t0_obs_valid = t0_event_obs[
            t0_valid
        ]

        t0_pred_valid = t0_event_pred[
            t0_valid
        ]

        hs_spearman = spearmanr(
            hs_obs_valid,
            hs_pred_valid,
        ).statistic

        t0_spearman = spearmanr(
            t0_obs_valid,
            t0_pred_valid,
        ).statistic

        metrics_hs = (
            f"RMSE: "
            f"{root_mean_squared_error(
                hs_obs_valid,
                hs_pred_valid,
            ):.2f} m\n"
            f"Bias: "
            f"{np.mean(
                hs_pred_valid
                - hs_obs_valid
            ):.2f} m\n"
            f"MAE: "
            f"{mean_absolute_error(
                hs_obs_valid,
                hs_pred_valid,
            ):.2f} m\n"
            f"R²: "
            f"{r2_score(
                hs_obs_valid,
                hs_pred_valid,
            ):.2f}\n"
            f"rₛ: {hs_spearman:.2f}\n"
            f"N: {n_hs}"
        )

        hs_metrics_box = AnchoredText(
            metrics_hs,
            loc="lower left",
            prop={
                "size": 5,
                "color": utils.hereon_cp[2],
            },
            frameon=False,
        )

        ax1.add_artist(
            hs_metrics_box
        )

        metrics_t0 = (
            f"RMSE: "
            f"{root_mean_squared_error(
                t0_obs_valid,
                t0_pred_valid,
            ):.2f} s\n"
            f"Bias: "
            f"{np.mean(
                t0_pred_valid
                - t0_obs_valid
            ):.2f} s\n"
            f"MAE: "
            f"{mean_absolute_error(
                t0_obs_valid,
                t0_pred_valid,
            ):.2f} s\n"
            f"R²: "
            f"{r2_score(
                t0_obs_valid,
                t0_pred_valid,
            ):.2f}\n"
            f"rₛ: {t0_spearman:.2f}\n"
            f"N: {n_t0}"
        )

        t0_metrics_box = AnchoredText(
            metrics_t0,
            loc="lower right",
            prop={
                "size": 5,
                "color": utils.hereon_cp[5],
            },
            frameon=False,
        )

        ax2.add_artist(
            t0_metrics_box
        )

    # ------------------------------------------------------------------
    # x-axis formatting
    # ------------------------------------------------------------------
    ax1.xaxis.set_major_locator(
        mdates.DayLocator(
            interval=7
        )
    )

    ax1.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b %d"
        )
    )

    # draw before retrieving final tick positions
    if fig is not None:
        fig.canvas.draw()
    else:
        ax1.figure.canvas.draw()

    ticks = ax1.get_xticks()

    labels = [
        mdates.num2date(
            tick
        ).strftime("%b %d")
        for tick in ticks
    ]

    if labels:
        labels[-1] = mdates.num2date(
            ticks[-1]
        ).strftime(
            "%b %d, %Y"
        )

        ax1.set_xticks(
            ticks
        )

        ax1.set_xticklabels(
            labels
        )

    # ------------------------------------------------------------------
    # combined legend
    # ------------------------------------------------------------------
    handles_1, labels_1 = (
        ax1.get_legend_handles_labels()
    )

    handles_2, labels_2 = (
        ax2.get_legend_handles_labels()
    )

    ax1.legend(
        handles_1 + handles_2,
        labels_1 + labels_2,
        loc="upper right",
        fontsize=6,
        ncol=2,
        frameon=False,
    )

    if fig is not None:
        fig.tight_layout()

    return fig, ax1, ax2


# def plot_storm(data_obs,data_pred, date1,date2):
#     wave_data_obs = data_obs
#     wave_data_pred = data_pred
#     fig, axs = plt.subplots(1, 1, figsize=(4.2, 2), dpi=600)
#     # Plot the wave height
#     axs.plot(wave_data_obs.loc[date1:date2,'hs'], lw=0.4, color=utils.hereon_cp[2], label = 'observed', linestyle='--') #, marker='.',  markersize=.3
#     axs.plot(wave_data_pred.loc[date1:date2,'hs'], lw=0.4, color=utils.hereon_cp[2], label = 'prediction', linestyle='-')
    
#     # Threshold lines
#     # axs.axhline(1.48, lw=0.2, color='gray', linestyle='--')
#     # axs.axhline(2.48, lw=0.2, color='gray', linestyle='--')
    
#     # Customize axes
#     axs.spines['top'].set_visible(False)
#     axs.spines['right'].set_visible(False)
#     axs.yaxis.tick_left()
#     axs.xaxis.tick_bottom()
#     axs.tick_params(which='minor', bottom=False, left=False)
#     axs.set_ylim(0.1,7.9)
#     # axs.set_xticklabels([])
#     # Format the x-axis to show dates
#     axs.xaxis.set_major_locator(mdates.DayLocator(interval=7))
#     axs.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
#     # Get current tick positions and labels
#     ticks = axs.get_xticks()
#     labels = [mdates.num2date(tick).strftime('%b %d') for tick in ticks]
    
#     # Replace the last label with year info
#     if len(labels) > 0:
#         labels[-1] = mdates.num2date(ticks[-1]).strftime('%b %d, %Y')
#         axs.set_xticklabels(labels)
    
#     axs.set_ylabel('Significant wave height, H$_s$ [m]')
#     # Add legend with custom handles
#     axs.legend(loc='upper right', 
#               fontsize=6, 
#               frameon=False)
#     # Save and display the figure
#     axs.grid(linewidth = 0.05, which='major', axis='y')
#     fig.tight_layout()
