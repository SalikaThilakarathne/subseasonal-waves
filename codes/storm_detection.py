#(c) Salika Thilakarathne, Hereon.
#03 December, 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utils

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
                current['max_hs'] = max(current['max_hs'], next_event['max_hs'])
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

def plot_storm(data_obs, data_pred, date1, date2, ax=None, figsize=(4.2, 2.2), dpi=600, fig_num=""):
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
