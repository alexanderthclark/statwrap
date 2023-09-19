'''
Stats functions adapted to the conventions of Freedman, Pisani, and Purves 2007.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.magic import register_line_magic
from statwrap.utils import modify_std, args_to_array

def average(*args):
	"""
    Computes the arithmetic mean.

    Parameters
    -----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both average([1,2]) and average(1,2) are valid.

    Returns
    --------
    float
        The average value, or arithmetic mean, for a collection of numbers.

    Example
    --------
    >>> average(0, 5, -8, 7, -3)
    0.2

	"""
	a = args_to_array(args)
	return np.mean(a)

def rms_size(*args):
	"""
    Computes the r.m.s. (Root Mean Square) size of a list of numbers.

    Parameters
    -----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both rms_size([1,2]) and rms_size(1,2) are valid.

    Returns
    --------
    float
        The r.m.s. value of the provided numbers.

    Example
    --------
    >>> rms_size(0, 5, -8, 7, -3)
    5.422176684690384

	"""
	a = args_to_array(args)
	squared = [r**2 for r in a]
	return np.sqrt(np.mean(squared))

def sd(*args):
    """
    Computes the population standard deviation, or SD.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both sd([1,2]) and sd(1,2) are valid.

    Returns
    -------
    float
        The population standard deviation of the input array.

    Examples
    --------
    >>> sd([-1, 0, 1])
    0.816496580927726

    >>> sd(-1,0,1)
    0.816496580927726

    """
    a = args_to_array(args)
    return np.std(a, ddof=0)

def var(*args):
    """
    Computes the population variance.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both var([1,2]) and var(1,2) are valid.

    Returns
    -------
    float
        The population variance of the input array.

    Examples
    --------
    >>> var([-1, 0, 1])
    0.6666666666666666

    >>> var(-1, 0, 1)
    0.6666666666666666

    """
    a = args_to_array(args)
    return np.var(a, ddof=0)

def sd_plus(*args):
    """
    Computes the sample standard deviation, or SD+.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both sd_plus([1,2]) and sd_plus(1,2) are valid.

    Returns
    -------
    float
        The sample standard deviation of the input array.

    Examples
    --------
    >>> sd_plus([-1, 0, 1])
    1.0

    >>> sd_plus(-1, 0, 1)
    1.0

    """
    a = args_to_array(args)
    return np.std(a, ddof=1)

def var_plus(*args):
    """
    Computes the sample variance.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both var_plus([1,2]) and var_plus(1,2) are valid.

    Returns
    -------
    float
        The sample variance of the input array.

    Examples
    --------
    >>> var_plus([-1, 0, 1])
    1.0

    >>> var_plus(-1, 0, 1)
    1.0

    """
    a = args_to_array(args)
    return np.var(a, ddof=1)

def change_std_behavior(pd_obj):
    original = getattr(pd_obj, 'std')
    pop_std, sample_std = modify_std(original)
    setattr(pd_obj, 'std', pop_std)
    setattr(pd_obj, 'sd', pop_std)
    setattr(pd_obj, 'sd_plus', sample_std)

def apply_pd_changes():
    change_std_behavior(pd.DataFrame)
    change_std_behavior(pd.Series)

def histogram(x, bins=None, density=False, xlim=None, ylim=None, ax=None, show=True, save_as=None, **kwargs):
    '''
    Creates a histogram using matplotlib.

    Parameters
    ----------
    x : array-like or sequence or array-likes
        Input data to be plotted as a histogram.

    bins : int or sequence, optional
        The number of bins or the bin edges if a sequence is provided. If not provided, defaults are used.

    density : bool, default False
        If True, normalizes the histogram so that the total area is equal to 1.

    xlim : tuple, optional
        The x-axis limits as (min, max). If not provided, defaults are used.

    ylim : tuple, optional
        The y-axis limits as (min, max). If not provided, defaults are used.

    ax : matplotlib axes object, optional
        An existing axes to draw the histogram on. If None, a new figure and axes are created.

    show : bool, default True
        If True, displays the plot. Otherwise, it returns the figure and axis.

    save_as : str, optional
        If a string is provided, the figure is saved with the given filename.
        This must include an extension like '.png' or '.pdf'.

    kwargs : dict
        Additional keyword arguments to pass to `ax.hist`.

    Returns
    -------
    fig, ax : tuple
        A tuple containing the figure and axis objects. Only returned if `show` is False.

    Example
    --------
    >>> histogram([1,2,3,3,3], save_as = 'example.png')
    (histogram will appear in notebook output)

    '''
    if ax is None:
        fig, ax = plt.figure(), plt.axes()

    # Documented parameters
    kwargs['bins'] = bins
    kwargs['density'] = density

    # I like black edges
    if ('edgecolor' not in kwargs) and ('ec' not in kwargs):
            kwargs['ec'] = 'black'

    ax.hist(x, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    else:
        return fig, ax

def fpp_setup():
    apply_pd_changes()
