from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_xy(
    x: Sequence[float],
    y: Sequence[float] | Sequence[Sequence[float]],
    legends: Sequence[str] | None = None,
    *,
    title: str | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: tuple[float, float] = (8.0, 5.0),
    styles: Sequence[str] | None = None,
    linewidth: float = 2.0,
    marker: str | None = None,
    grid: bool = True,
    legend_loc: str = "best",
    tight_layout: bool = True,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Generic XY plotting utility for single or multiple y-series.

    Examples
    --------
    plot_xy(t, reactor_volume, legends=["V"])
    plot_xy(t, [na, nb, nc], legends=["A", "B", "C"], xlabel="t (s)", ylabel="n (mol)")
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)

    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    elif y_arr.ndim != 2:
        raise ValueError("y must be a 1D series or 2D array-like (n_series, n_points).")

    if y_arr.shape[1] != x_arr.size:
        if y_arr.shape[0] == x_arr.size:
            y_arr = y_arr.T
        else:
            raise ValueError(
                f"Incompatible shapes: x has {x_arr.size} points, y has shape {y_arr.shape}."
            )

    n_series = y_arr.shape[0]

    if legends is None:
        legends = [f"series_{i+1}" for i in range(n_series)]
    if len(legends) != n_series:
        raise ValueError("len(legends) must match the number of y series.")

    if styles is not None and len(styles) != n_series:
        raise ValueError("len(styles) must match the number of y series.")

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_series):
        line_style = "-" if styles is None else styles[i]
        ax.plot(
            x_arr,
            y_arr[i, :],
            line_style,
            linewidth=linewidth,
            marker=marker,
            label=legends[i],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if legends:
        ax.legend(loc=legend_loc)
    if tight_layout:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax
