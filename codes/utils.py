# ---------------------------------------------------------------------------
# a collection of small Python functions for the transformer encoder deep learning model
# Salika Thilakarathne
# Helmholtz-Zentrum Hereon
# March, 2026
# ---------------------------------------------------------------------------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score

from shapely.geometry import LineString
from shapely.geometry import Point #for shoreline detection
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import pearsonr
import scipy.stats as stats


import warnings
import typing
import numpy as np
from packaging.version import Version
from packaging.specifiers import SpecifierSet
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf

############################################################
##### Fig default setting
############################################################


#HEREON colors palate
hereon_cp = sns.color_palette(['#0091A0','#E60046','#00AAE6','#FA7350','#FAB423','#8C5AB4','#595959','#C5FFFD','#88D9E6']) 
#old gray color:#7F7F7F
#to create figures--better
import scienceplots
def scientific_graphs():
    """
    Set default styles for scientific graphs, including colors for spines, labels, and ticks.
    """
    # Reset to default settings
    plt.rcdefaults()
    # Update global parameters
    plt.rcParams.update({
        'font.size': 9,               # Default font size
        'mathtext.default': 'regular', # Use regular math font
        'axes.labelcolor': hereon_cp[6], # Default label color
        'xtick.color': hereon_cp[6],     # X-axis tick color
        'ytick.color': hereon_cp[6],     # Y-axis tick color
        'axes.edgecolor': hereon_cp[6],  # Spine color
        'figure.autolayout': True,        # Enable tight layout by default
        # 'figure.dpi': 300,                # Default DPI for figures shown
        # 'savefig.dpi': 300,               # Default DPI for saved figures

    })
    # Apply a scientific style
    plt.style.use(['science', 'no-latex', 'nature'])
# Call the function to apply the defaults
scientific_graphs()

#if want to shift to defaluts
def default_graphs():
    plt.rcdefaults()
    plt.rcParams.update({'font.size': 9})    

############################################################
##### Utility Functions for calculations etc -- mostly for Hasaki, Japan datasets
############################################################
#to flattern a data series
def flatten_st(data):
    return pd.Series(data).values.flatten()

#define a function to take a sample: for a given bin it is the min of sample wanted or the data size...
def sample(data, size):
    return data.sample(min(len(data),size), random_state = 42) 

def stat_obs_pred(obs,pred):
    print(f"MAE: {mean_absolute_error(obs, pred):.2f}")
    print(f"MSE: {mean_squared_error(obs, pred):.2f}")
    print(f"R²: {r2_score(obs, pred):.2f}")
    print(f"r: {pearsonr(obs, pred)[0]:.2f}")

def compute_quantile_stats(obs, pred, q=4):
    """
    Compute bias, std and Spearman correlation for user-defined quantile bins.
    Parameters
    ----------
    obs : pandas.Series, Observed values.
    pred : pandas.Series, Predicted values.
    q : int, Number of quantiles (e.g., 4, 5, 10).  
    Returns
    -------
    pandas.DataFrame, Statistics per quantile.
    """
    
    df = pd.DataFrame({"obs": obs, "pred": pred})
    # quantile labels: Q1, Q2, ..., Qn
    labels = [f"Q{i}" for i in range(1, q + 1)]
    
    df["obs_q"] = pd.qcut(
        df["obs"], 
        q=q, 
        labels=labels,
        duplicates="drop")
    # bias
    df["bias"] = df["pred"] - df["obs"]
    
    # stats by quantile
    stats_by_q = df.groupby("obs_q").apply(
        lambda g: pd.Series({
            "bias_mean": g["bias"].mean(),
            "std_obs": g["obs"].std(ddof=1),
            "std_pred": g["pred"].std(ddof=1),
            "spearman_corr": g["obs"].corr(g["pred"], method="spearman")
        }),
        include_groups=False)
    return stats_by_q

def bin_wave_directions(wave_dir, bin_size=15):
    """
    Bins wave direction data into intervals and returns counts per bin,
    labeling each bin by its center (mean) direction.

    Parameters:
        wave_dir (array-like): Series or array of wave directions in degrees (0–360).
        bin_size (int): Size of each bin in degrees (default: 15).

    Returns:
        pd.DataFrame: DataFrame with bin centers and counts.
    """
    wave_dir = np.asarray(wave_dir) % 360  # Ensure directions are in 0–360
    bins = np.arange(0, 361, bin_size)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    # Bin the data
    binned = pd.cut(wave_dir, bins=bins, right=False, include_lowest=True, labels=bin_centers)
    bin_counts = binned.value_counts().sort_index()

    return pd.DataFrame({
        'bin_cen_deg': bin_counts.index.astype(float),
        'counts': bin_counts.values
    })


############################################################
##### Utility Functions for differnt types of figures
############################################################



def plot_predictions(y_true, y_pred, loc=4, 
                    scatter_color='white', edge_color=hereon_cp[0], marker_size=6,
                    fig_num='', xlabel='Observed value', ylabel='Predicted value', lim1=1, lim2 = 3,
                    text_color=hereon_cp[6], figsize=(2.1, 2), dpi=450, ax=None):
    """
    Create a customised scatter plot comparing true values vs predictions.
    """
    # Use provided axis or create new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = None  # Will return None if user is embedding it

    # Scatter plot
    ax.scatter(y_true, y_pred, 
               color=scatter_color, 
               s=marker_size, 
               edgecolor=edge_color,
               linewidths=0.15,
               alpha=1,
               zorder=1)

    # Diagonal reference line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val-1, max_val+1], [min_val-1, max_val+1], 
            '-', color=hereon_cp[1], lw=0.4, zorder=2)

    # Labels and grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(lim1, lim2)
    ax.set_xlim(lim1, lim2)
    ax.grid(lw=0.1)

    # ax.set_xticks([1.2, 1.6, 2.0, 2.4])
    # ax.set_yticks([1.2, 1.6, 2.0, 2.4])

    # Subfigure label
    ax.text(0.54, 0.95, fig_num, transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color=hereon_cp[6])

    # Metrics box
    metrics_text = (
        f"RMSE: {root_mean_squared_error(y_true, y_pred):.2f}\n" #root_mean_squared_error, f"MSE: {mean_squared_error(y_true, y_pred):.2f}\n"
        f"MAE: {mean_absolute_error(y_true, y_pred):.2f}\n"
        f"R²: {r2_score(y_true, y_pred):.2f}\n"
        f"rₛ: {np.corrcoef(y_true, y_pred)[0,1]:.2f}"
    )
    at = AnchoredText(metrics_text,
                      loc=loc,
                      prop=dict(size=5, color=text_color),
                      frameon=False,
                      bbox_transform=ax.transData)
    ax.add_artist(at)

    if fig is not None:
        plt.tight_layout()
    return fig, ax

def plot_predictions_with_errors(y_true, y_pred, loc=4,
                                 scatter_color='none', edge_color='k',
                                 xticks=[0, 1], yticks=[0, 1],
                                 xlabel='Observed values', ylabel='Predicted values',
                                 text_color=hereon_cp[6], figsize=(2.5, 5), dpi=600):
    """
    Create a customized scatter plot comparing true values vs predictions,
    and a subplot showing the probability distribution of errors.
    """
    # Create a figure with two subplots, stacked vertically.
    # The first subplot is for the scatter plot, the second is for the error distribution.
    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi, gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)

    # --- Scatter Plot Subplot (axs[0]) ---
    
    # Scatter plot with thin edges
    axs[0].scatter(y_true, y_pred,
                   color=scatter_color,
                   s=10,
                   edgecolor=edge_color,
                   linewidths=0.15,
                   zorder=2)

    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axs[0].plot([min_val, max_val], [min_val, max_val],
                'k--', lw=0.4, zorder=1)

    # Labels and grid for the scatter plot
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(lw=0.1)

    # Metrics text
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = np.corrcoef(y_true, y_pred)[0,1]
    metrics_text = (
        f"MAE: {mae:.2f}\n"
        f"MSE: {mse:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"r: {r:.2f}"
    )

    # Add anchored text
    at = AnchoredText(metrics_text,
                      loc=loc,
                      prop=dict(size=5, color=text_color),
                      frameon=False,
                      bbox_transform=axs[0].transData)
    axs[0].add_artist(at)
    axs[0].set_xticks(xticks)
    axs[0].set_yticks(yticks)
    axs[0].set_aspect('equal', adjustable='box')

    # --- Error Distribution Subplot (axs[1]) ---

    # Calculate the errors
    errors = y_true - y_pred

    # Plot histogram of the errors
    axs[1].hist(errors, bins=30, density=True, color=hereon_cp[8], label='Histogram')

    # Plot the probability distribution (KDE)
    density = stats.gaussian_kde(errors)
    x_vals = np.linspace(min(errors), max(errors), 200)
    axs[1].plot(x_vals, density(x_vals), color=hereon_cp[2], linewidth=0.4, label='PDF')

    axs[1].axvline(0, color='r', linestyle='-', linewidth=0.4, label='Zero Error')

    axs[1].set_xlabel('Error ($y_{true} - y_{pred}$)')
    axs[1].set_ylabel('Probability Density')
    axs[1].grid(lw=0.1)
    axs[1].set_xticks([-1,0,1])
    axs[1].legend(fontsize=6)

    # plt.tight_layout()
    return fig, axs

def plot_sub_predictions(y_true, y_pred, loc=4, 
                     scatter_color='white', edge_color='k',
                     text_color=hereon_cp[6],
                     figsize=(2.1, 2), dpi=300, ax=None,min_val=0, max_val=50):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.scatter(y_true, y_pred, 
               color=scatter_color, 
               s=6, 
               edgecolor=edge_color,
               linewidths=0.2,
               zorder=2)

    # min_val = min(y_true.min(), y_pred.min())
    # max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'k--', lw=0.4, zorder=1)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel('Observed values')
    ax.set_ylabel('Predicted values')
    ax.grid(lw=0.1)
    r_value, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    metrics_text = (
        f"MAE: {mean_absolute_error(y_true, y_pred):.2f}\n"
        f"MSE: {mean_squared_error(y_true, y_pred):.2f}\n"
        f"R²: {r2_score(y_true, y_pred):.2f}\n"
        f"r: {r_value:.2f}" 
)
    
    at = AnchoredText(metrics_text,
                      loc=loc,
                      prop=dict(size=5, color=text_color),
                      frameon=False,
                      bbox_transform=ax.transData)
    ax.add_artist(at)
    return fig, ax

def plot_history(history, key, save_path=None):
    fig, axs = plt.subplots(1, 1, figsize=(4.2, 2.0), dpi=600)
    
    axs.plot(history.history[key], linewidth=0.5, label=f'Train {key}')
    axs.plot(history.history['val_' + key], linewidth=0.5, label=f'Val {key}')
    
    axs.set_xlabel("Epochs")
    axs.set_ylabel(key.replace('_', ' ').title())
    axs.grid(linewidth=0.1, alpha=0.5)
    
    # Minimalist style: remove top and right spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.yaxis.tick_left()
    axs.xaxis.tick_bottom()
    
    axs.legend(frameon=False)
    
    if save_path:
        plt.savefig(save_path, dpi=450, bbox_inches='tight')


def plot_observed_vs_predicted(wave_data, test_predict, features, val_end, dates, colors, save_path=None):
    """
    Create and plot a time series line graph comparing observed and predicted values.

    Parameters:
    - wave_data: Original DataFrame with index as datetime and observed values.
    - test_predict: Numpy array of predicted values.
    - features: List of features (single target assumed).
    - val_end: Index where testing data starts in wave_data.
    - dates: Tuple/list of date range for plotting (start, end).
    - colors: Tuple/list with two colors [observed_color, predicted_color].
    - save_path: Optional path to save the plot.
    """
    # Create dataframe for plotting
    temp_df = pd.DataFrame({
        'date': wave_data.iloc[val_end:].index,
        'observed': wave_data.iloc[val_end:][features].values.ravel(),
        'predicted': test_predict[:, 0]
    }).set_index('date')

    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(6.2, 1.5), dpi=300)
    subset = temp_df.loc[dates[0]:dates[1]]
    sns.lineplot(data=subset, x=subset.index, y='observed', label='Observed', color=colors[0], lw=0.4)
    sns.lineplot(data=subset, x=subset.index, y='predicted', label='Predicted', color=colors[1], lw=0.4)

    plt.xlabel('')
    plt.ylabel('Surge height (m)')
    plt.grid(linewidth=0.1)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.yaxis.tick_left()
    axs.xaxis.tick_bottom()
    plt.legend(frameon=False)

    # Save if requested
    if save_path:
        plt.savefig(f"{save_path}/results_time-series-performance_{features[0]}.png", dpi=450, bbox_inches='tight')


"""Taylor Diagrams."""

class TaylorDiagram:
    """Create a Taylor diagram for comparing model vs reference statistics."""

    def __init__(
        self,
        refstd: float = 1,
        fig: matplotlib.figure.Figure = None,
        rect: int = 111,
        label: str = 'REF',
        std_range: tuple = (0, 1.65),
        std_level: list = np.arange(0, 1.51, 0.25),
    ):
        self.fig = fig or plt.figure(figsize=(8, 8))
        self.refstd = refstd
        self.smin, self.smax = std_range

        mpl_version = Version(matplotlib.__version__)
        if mpl_version in SpecifierSet("<3.9", prereleases=True):
            tr = PolarAxes.PolarTransform(_apply_theta_transforms=False)
        elif mpl_version in SpecifierSet("<3.11", prereleases=True):
            tr = PolarAxes.PolarTransform(apply_theta_transforms=False)
        else:
            tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.concatenate((np.arange(10) / 10.0, [0.95, 0.99, 1]))
        tlocs = np.arccos(rlocs)
        gl1 = gf.FixedLocator(tlocs)
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Stddev labels
        gl2 = gf.FixedLocator(std_level)
        labels = list(map(lambda x: f"{x:0.2f}", std_level))
        labels[np.where(std_level == self.refstd)[0][0]] = label
        tf2 = gf.DictFormatter(dict(zip(std_level, labels)))

        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi / 2, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
            grid_locator2=gl2,
            tick_formatter2=tf2,
        )

        ax = fa.FloatingSubplot(self.fig, rect, grid_helper=ghelper)
        self.fig.add_subplot(ax)

        # Axes labels
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["right"].label.set_text("Standard deviation (Normalized)")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Reference stddev contour
        t = np.linspace(0, np.pi / 2)
        self.referenceLine, = self.ax.plot(
            t, np.zeros_like(t) + self.refstd, linestyle=(0, (9, 5)), color='black'
        )

        self.ax.set_aspect('equal')
        self.modelMarkerSet = []
        self.modelOutside = -1

    # ------------------------------------------------------------------

    def add_model_set(
        self,
        stddev,
        corrcoef,
        fontsize: float = 14,
        xytext: tuple = (-5, 7),
        annotate_on: bool = True,
        model_outlier_on: bool = False,
        percent_bias_on: bool = False,
        bias_array=None,
        *args,
        **kwargs,
    ):
        """Add points for a model set."""
        std = np.array(stddev)
        corr = np.array(corrcoef)
        std_plot, corr_plot = std, corr

        index_map = dict(zip(std, range(1, len(std) + 1)))

        if percent_bias_on:
            if bias_array is None:
                raise AttributeError("bias_array required when percent_bias_on=True")
            if kwargs.get("s") or kwargs.get("marker"):
                raise AttributeError("Do not supply s/marker with percent_bias_on=True")
            bias = np.array(bias_array)

        if model_outlier_on:
            cond = np.logical_and(std <= self.smax, corr >= self.smin)
            std_plot, corr_plot = std[cond], corr[cond]
            std_out, corr_out = std[~cond], corr[~cond]
            if percent_bias_on:
                bias, bias_out = bias[cond], np.array(bias_array)[~cond]

        # Markers
        if not percent_bias_on:
            modelset = self.ax.scatter(np.arccos(corr_plot), std_plot, *args, **kwargs)
        else:
            for i in range(len(corr_plot)):
                size, mk = self._bias_to_marker_size(bias[i])
                self.ax.scatter(
                    np.arccos(corr_plot)[i], std_plot[i], s=size, marker=mk, *args, **kwargs
                )
            color = kwargs.get("color") or kwargs.get("edgecolors") or kwargs.get("facecolors")
            modelset = plt.scatter(15, 25, color=color, label=kwargs.get("label"))

        self.modelMarkerSet.append(modelset)
        texts = []

        if annotate_on:
            for s, c in zip(std_plot, corr_plot):
                txt = self.ax.annotate(
                    str(index_map[s]),
                    (np.arccos(c), s),
                    fontsize=fontsize,
                    color=kwargs.get("color"),
                    textcoords="offset pixels",
                    xytext=xytext,
                )
                texts.append(txt)

        if model_outlier_on and len(std_out) > 0:
            for s, c in zip(std_out, corr_out):
                self.modelOutside += 1
                x = 0.054 + self.modelOutside * 0.22
                y = -0.105
                self.ax.scatter(x, y, *args, **kwargs, clip_on=False, transform=self.ax.transAxes)
                texts.append(
                    self.ax.text(
                        x - 0.01, y + 0.025, str(index_map[s]), fontsize=fontsize, transform=self.ax.transAxes
                    )
                )
                self.ax.text(
                    x + 0.03, y,
                    rf'$\frac{{{s:.2f}}}{{{c:.2f}}}$',
                    fontsize=17,
                    transform=self.ax.transAxes,
                )

        return texts, modelset

    # ------------------------------------------------------------------

    def add_corr_grid(self, arr, color='lightgray', linestyle=(0, (9, 5)), linewidth=0.5, **kwargs):
        """Vertical correlation gridlines."""
        for v in arr:
            self.ax.vlines(np.arccos(v), ymin=self.smin, ymax=self.smax,
                           color=color, linestyle=linestyle, linewidth=linewidth, **kwargs)

    def add_std_grid(self, arr, color='lightgray', linestyle=(0, (9, 5)), linewidth=1, **kwargs):
        """Radial stddev gridlines."""
        t = np.linspace(0, np.pi / 2)
        for v in arr:
            self.ax.plot(t, np.zeros_like(t) + v, color=color,
                         linestyle=linestyle, linewidth=linewidth, **kwargs)

    # ------------------------------------------------------------------

    def add_contours(self, levels=5, **kwargs):
        """Add RMS difference contours."""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, np.pi / 2))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts))
        return self.ax.contour(ts, rs, rms, levels, **kwargs)

    def add_model_name(self, namearr, x_loc=0.1, y_loc=0.31, fontsize=13, **kwargs):
        """List model names in axes coordinates."""
        text = '\n'.join([f"{i+1} - {namearr[i]}" for i in range(len(namearr))])
        self.ax.text(x_loc, y_loc, text, fontsize=fontsize, transform=self.ax.transAxes, **kwargs)

    def add_bias_legend(self, y_offset=0.0, fontsize=11):
        """Bias legend."""
        text = "- / + Bias\n"
        self.ax.text(0.07, 0.92 + y_offset, text, fontsize=fontsize,
                     verticalalignment="top", transform=self.ax.transAxes)

        percent = [">20%", "10-20%", "5-10%", "1-5%", "<1%"]
        size = [130, 90, 50, 30, 60]
        y = 0.85 + y_offset

        for i in range(5):
            m1, m2 = ("v", "^") if i < 4 else ("o", "o")

            self.ax.scatter(0.08, y, s=size[i], marker=m1, edgecolors='black',
                            facecolors="none", linewidths=0.5, transform=self.ax.transAxes)
            self.ax.scatter(0.13, y, s=size[i], marker=m2, edgecolors='black',
                            facecolors="none", linewidths=0.5, transform=self.ax.transAxes)

            self.ax.text(0.18, y - 0.01, percent[i], fontsize=fontsize,
                         transform=self.ax.transAxes)
            y -= 0.05

    def add_legend(self, xloc=1.1, yloc=0.95, loc="upper right", fontsize=14, **kwargs):
        handles = self.modelMarkerSet[::-1]
        labels = [p.get_label() for p in handles]
        return self._ax.legend(handles, labels, loc=loc,
                               bbox_to_anchor=(xloc, yloc), fontsize=fontsize, frameon=False)

    def add_title(self, text, fontsize=18, y_loc=None, **kwargs):
        self._ax.set_title(text, fontsize=fontsize, y=y_loc, **kwargs)

    def set_fontsizes_and_pad(self, ticklabel_fontsize=14, axislabel_fontsize=16, axislabel_pad=8):
        self._ax.axis['top', 'right', 'left'].major_ticklabels.set_fontsize(ticklabel_fontsize)
        self._ax.axis['top', 'right'].label.set_fontsize(axislabel_fontsize)
        self._ax.axis['top', 'right'].label.set_pad(axislabel_pad)

    # ------------------------------------------------------------------

    def _bias_to_marker_size(self, bias):
        """Internal: return marker size + shape from bias value."""
        ab = abs(bias)
        if ab > 20: s = 130
        elif ab > 10: s = 90
        elif ab > 5: s = 50
        elif ab > 1: s = 30
        else: s = 60

        if ab <= 1: m = 'o'
        elif bias > 0: m = '^'
        else: m = 'v'
        return s, m