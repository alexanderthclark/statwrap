"""Shared plotting helpers used across StatWrap conventions."""

import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(
    x,
    y,
    xlim=None,
    ylim=None,
    ax=None,
    show=True,
    save_as=None,
    xlabel=None,
    ylabel=None,
    title=None,
    regression_line=False,
    regression_equation=False,
    **kwargs,
):
    """
    Create a scatter plot of `x` versus `y`, with specified axis labels, limits,
    title, and other properties. Optionally, a regression line can be added to
    the plot.

    Parameters
    ----------
    x : array-like
        The data values for the x-axis.
    y : array-like
        The data values for the y-axis.
    xlim : tuple, optional
        The limits for the x-axis in the form of (xmin, xmax). Default is None.
    ylim : tuple, optional
        The limits for the y-axis in the form of (ymin, ymax). Default is None.
    ax : matplotlib.axes._axes.Axes, optional
        The axes upon which to plot. If None, new axes will be created. Default
        is None.
    show : bool, optional
        If True, display the plot. If False, return the figure and axes without
        displaying them. Default is True.
    save_as : str, optional
        The filename (with path) to save the figure. If None, the figure is not
        saved. Default is None.
    xlabel : str, optional
        The label for the x-axis. Default is None.
    ylabel : str, optional
        The label for the y-axis. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    regression_line : bool, optional
        If True, a regression line will be added to the plot. Default is False.
    regression_equation : bool, optional
        If True, the regression equation will be added above the plot. Default
        is False.
    **kwargs : dict
        Additional keyword arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes._axes.Axes
        The figure and axes objects, returned only if `show` is False.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(50)
    >>> y = np.random.rand(50)
    >>> scatter_plot(
    ...     x,
    ...     y,
    ...     xlabel='X-axis',
    ...     ylabel='Y-axis',
    ...     title='Scatter Plot',
    ...     regression_line=True,
    ... )
    """
    if ax is None:
        _, ax = plt.subplots()

    if "alpha" not in kwargs and len(x) > 100:
        kwargs["alpha"] = 0.5

    x = np.squeeze(np.asarray(x))
    y = np.squeeze(np.asarray(y))
    ax.scatter(x, y, **kwargs)

    slope, intercept = np.polyfit(x, y, 1)

    if regression_line:
        ax.plot(x, slope * x + intercept, color="gray")

    if regression_equation:
        equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
        ax.text(
            0.5,
            1,
            equation_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="center",
            alpha=0.5,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        pad = 12 if regression_equation else None
        ax.set_title(title, pad=pad)

    if save_as is not None:
        ax.figure.savefig(save_as)
    if show:
        plt.show()
    else:
        return ax.figure, ax


__all__ = ["scatter_plot"]
