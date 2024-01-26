'''
Stats functions adapted to the conventions of Freedman, Pisani, and Purves 2007.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.magic import register_line_magic
from statwrap.utils import modify_std, args_to_array, formula

def box_model(*args, with_replacement = True, draws = 1, random_seed = None):
    """
    Returns random draws from a box model where each number in the box model
    is equally likely to be drawn.

    Parameters
    ----------
    *args : tuple or array_like
        The elements forming the box from which numbers will be drawn.
        If a single array_like is provided, it will be used as the box model.
        If multiple values are provided, they should be passed as a flat tuple.
    with_replacement : bool, optional
        Specifies whether drawing is done with replacement. Default is True,
        where numbers are replaced back into the box after each draw.
    draws : int, optional
        The number of draws to be made from the box. Default is 1.
    random_seed : int, optional
        The seed for the random number generator ensuring reproducibility of the
        random draws. By default, none is passed.

    Returns
    -------
    single value or list
        If `draws` is 1, returns a single value from the box. If `draws` is greater than 1,
        returns a list of length `draws`, containing the randomly drawn numbers from the box.

    Examples
    --------
    >>> box_model([1,2,3,4,5,6], with_replacement=True, draws=3)
    array([2, 5, 5])

    >>> box_model((1,2,3,4,5,6), with_replacement=False, draws=3)
    array([4, 2, 6])
    """
    a = args_to_array(args)
    if random_seed:
        rng = np.random.default_rng(random_seed)
        X = rng.choice(a, replace=with_replacement, size=draws)
    else:
        X = np.random.choice(a, replace=with_replacement, size=draws)
    if draws == 1:
        return X[0]
    else:
        return X.tolist()

def scatter_plot(x, y, xlim=None, ylim=None,
              ax=None, show=True, save_as=None, xlabel=None,
              ylabel=None, title=None, regression_line=False, **kwargs):
    """
    Create a scatter plot of `x` versus `y`, with specified axis labels, limits, title, and other properties.
    Optionally, a regression line can be added to the plot.

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
        The axes upon which to plot. If None, new axes will be created. Default is None.
    show : bool, optional
        If True, display the plot. If False, return the plot object without displaying it. Default is True.
    save_as : str, optional
        The filename (with path) to save the figure. If None, the figure is not saved. Default is None.
    xlabel : str, optional
        The label for the x-axis. Default is None.
    ylabel : str, optional
        The label for the y-axis. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    regression_line : bool, optional
        If True, a regression line will be added to the plot. Default is False.
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
    >>> scatter_plot(x, y, xlabel='X-axis', ylabel='Y-axis', title='Scatter Plot', regression_line=True)

    Notes
    -----
    If both `ax` and `show` are None, a new figure and axes will be created and displayed.
    """
    if ax is None:
        fig, ax = plt.figure(), plt.axes()

    if ('alpha' not in kwargs) and (len(x) > 100):
            kwargs['alpha'] = 0.5

    x = np.squeeze(np.array(x))
    y = np.squeeze(np.array(y))
    ax.scatter(x, y, **kwargs)

    if regression_line:
        m, b = np.polyfit(x, y, 1)  # Calculating the slope (m) and intercept (b) of the regression line
        ax.plot(x, m*x + b, color='gray')  # Plotting the regression line

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    else:
        return plt.gcf(), ax  # use gcf in case ax is passed in call

@formula
def r(x, y):
    """
    Calculates the Pearson correlation coefficient.

    .. math::
       \\frac{1}{n} \\sum_{i=1}^{n} \\dfrac{ (x_i - \\mu_x) (y_i - \\mu_y) }{ \\text{SD}_x \\times \\text{SD}_y }

    This is the average of the product of the z-scores.

    Parameters
    ----------
    x : array_like
        The first input array.

    y : array_like
        The second input array.

    Returns
    --------
    float
        The correlation coefficient.

    Example
    --------
    >>> r([0,1,1], [2,-9,2])
    -0.5

    """
    return np.corrcoef(x,y)[0][1]

@formula
def average(*args):
	"""
    Computes the arithmetic mean.

    .. math::
       \\frac{1}{n} \\sum_{i=1}^{n} x_i

    Parameters
    -----------
    args : array_like  or numeric scalars
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

@formula
def rms_size(*args):
	"""
    Computes the r.m.s. (Root Mean Square) size of a list of numbers.

    .. math::
       \\sqrt{ \\frac{1}{n}\\sum_{i=1}^{n}x_i^2 }

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

@formula
def sd(*args):
    """
    Computes the population standard deviation, or SD.

    .. math::
       \\sqrt{ \\frac{1}{n}\\sum_{i=1}^{n}(x_i-\mu)^2 }

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

@formula
def var(*args):
    """
    Computes the population variance.

    .. math::
       \\frac{1}{n}\\sum_{i=1}^{n}(x_i-\mu)^2

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

@formula
def sd_plus(*args):
    """
    Computes the sample standard deviation, or SD+.

    .. math::
       \\sqrt{ \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i-\\bar{x})^2 }

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

@formula
def var_plus(*args):
    """
    Computes the sample variance.

    .. math::
       \\frac{1}{n-1}\\sum_{i=1}^{n}(x_i-\\bar{x})^2

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

def histogram(*data_args, class_intervals=None, bins=None, density=True, xlim=None, ylim=None,
              ax=None, show=True, save_as=None, xlabel=None,
              ylabel=None, title=None, **kwargs):
    '''
    Creates a histogram using matplotlib.

    Parameters
    ----------
    data_args : array-like or sequence or array-likes or numeric scalars
        Input data to be plotted as a histogram.
    
    class_intervals : int or sequence, optional
        The number of blocks or the interval edges if a sequence is provided. If not provided, defaults are used.
    
    bins : int or sequence, optional
        Alternative name for class_intervals. class_intervals takes precedence is arguments are provided for both.

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

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    title : str, optional
        Title for the histogram plot.

    kwargs : dict
        Additional keyword arguments to pass to `ax.hist`.

    Returns
    -------
    fig, ax : tuple
        A tuple containing the figure and axis objects. Only returned if `show` is False.

    Examples
    --------
    >>> histogram([1,2,3,3,3], save_as = 'example.png')
    (histogram will appear in notebook output)

    >>> histogram(1,2,3,3,3, save_as = 'example.png')
    (alternate syntax producing the same histogram as above)

    >>> histogram([(1,2), (1,1,1,1)], title = 'Example')
    (overlapping histograms with two data sets)

    >>> histogram((1,2), (1,1,1,1), title = 'Example')
    (alternate syntax for overlapping histograms with two data sets)
    '''
    if ax is None:
        fig, ax = plt.figure(), plt.axes()

    # Documented parameters
    if class_intervals is not None:
        bins = class_intervals
    kwargs['bins'] = bins
    kwargs['density'] = density

    # I like black edges
    if ('edgecolor' not in kwargs) and ('ec' not in kwargs):
            kwargs['ec'] = 'black'

    x = args_to_array(data_args)
    ax.hist(x, **kwargs)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if density:
        y_ticks = ax.get_yticks()
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['{:.0f}%'.format(y*100) for y in y_ticks])

    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    else:
        return plt.gcf(), ax  # use gcf in case ax is passed in call

def fpp_setup():
    apply_pd_changes()
