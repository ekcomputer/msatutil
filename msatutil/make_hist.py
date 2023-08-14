from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Annotated


def make_hist(
    ax: plt.Axes,
    x: Sequence[float],
    label: str,
    color: str,
    rng: Optional[Annotated[Sequence[float], 2]] = None,
    nbins: Optional[int] = None,
    exp_fmt: bool = True,
):
    """
    Make a historgram of the data in x
    ax: matplotlib axes object
    x: array of data
    label: label for the legend
    color: color of the bars
    rng: range of the horizontal axis
    nbins: number of bins for the histogram
    """
    if rng is not None:
        x_rng = x[(x >= rng[0]) & (x <= rng[1])]
        x_mean = np.nanmean(x_rng)
        x_std = np.nanstd(x_rng, ddof=1)
        x_med = np.nanmedian(x_rng)
        if exp_fmt:
            label = (
                label
            ) = f"{label}\n$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}\nmedian: {x_med:.3e} "
        else:
            label = (
                label
            ) = f"{label}\n$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}\nmedian: {x_med:.2f} "
        bin_vals, bin_edges, patches = ax.hist(
            x,
            edgecolor=color,
            facecolor="None",
            label=label,
            range=rng,
            bins=nbins,
            histtype="step",
        )
    else:
        x_mean = np.nanmean(x)
        x_std = np.nanstd(x, ddof=1)
        x_med = np.nanmedian(x)
        if exp_fmt:
            label = f"{label}\n$\mu\pm\sigma$: {x_mean:.3e}$\pm${x_std:.3e}\nmedian: {x_med:.3e} "
        else:
            label = f"{label}\n$\mu\pm\sigma$: {x_mean:.2f}$\pm${x_std:.2f}\nmedian: {x_med:.2f} "
        bin_vals, bin_edges, patches = ax.hist(
            x,
            edgecolor=color,
            facecolor="None",
            label=label,
            histtype="step",
        )
    ax.axvline(x=x_med, color=color, linestyle="--")
    ax.legend(frameon=False)
    return np.max(bin_vals)
