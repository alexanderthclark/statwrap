'''
Stats functions adapted to the conventions of Google Sheets.
'''
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
from IPython.core.magic import register_line_magic
from statwrap.utils import (
    modify_std, 
    args_to_array, 
    hyperlink, 
    Hyperplane, 
    RegressionLine, 
)

@hyperlink
def linest(y, x, verbose = False):
    """
    Estimates a linear regression, akin to `LINEST() <https://support.google.com/docs/answer/3094249?hl=en>`_.

    This function performs a simple OLS (Ordinary Least Squares) regression
    on the provided input data to estimate the coefficients of a linear model.
    It returns a `Hyperplane` object representing the estimated linear model in
    the general case, or a `RegressionLine` object in the case of univariate 
    regression (i.e., when there's only one independent variable). These objects
    can be used to compute the predicted values of the dependent variable. If 
    `verbose` is set to True, it also returns a summary of the regression analysis.

    Parameters
    ----------
    y : array-like
        The dependent variable values. Should be a 1-dimensional array or list.
    x : array-like
        The independent variable values. Should be a list or a 2-dimensional array
        where each column represents a different variable.
    verbose : bool, optional
        If True, returns a detailed summary of the regression analysis along with
        the model object. Default is False.

    Returns
    -------
    Hyperplane or RegressionLine, (Optional) Regression Summary
        An object representing the estimated linear model. The coefficients 
        of the model are stored in the `coefficients` attribute of the 
        returned object, and the model can be called as a function to compute
        predicted values. In the case of univariate regression, a 
        `RegressionLine` object is returned. If `verbose` is True, also returns
        a summary of the regression analysis.

    Examples
    --------
    >>> y = [1, 2, 3, 4, 5]
    >>> x = [3, 4, 5, 6, 7]
    >>> model = linest(y, x)
    >>> model.coefficients
    array([-2., 1.])
    >>> model(10)  # prediction
    8
    >>> model_verbose = linest(y, x, verbose=True)
    >>> model_verbose[1]  # regression summary
    """
    try:
        x = pd.DataFrame(x)
        for idx, v in x.var().items():
            if v == 0:
                del x[idx]
    except:
        pass
    X = sm.add_constant(x)
    y = np.array(y)
    results = sm.OLS(y, X).fit()
    p = RegressionLine(y, x, results)
    
    if verbose:
        return p, results.summary()
    else:
        return p


@hyperlink
def normdist(x, mean=0, standard_deviation=1, cumulative=True):
    """
    Returns the value of the normal probability density function or cumulative distribution function.
    This mimics `NORMDIST() <https://support.google.com/docs/answer/3094021?hl=en&sjid=1926096259077083635>`_.

    Parameters
    ----------
    x : float or array-like
        The point(s) at which to evaluate the distribution.
    mean : float, optional
        The mean of the Normal distribution. Default is 0.
    standard_deviation : float, optional
        The standard deviation of the Normal distribution. Default is 1.
    cumulative : bool, optional
        Whether to compute the cumulative distribution function (CDF) or the 
        probability density function (PDF). Default is True (CDF).

    Returns
    -------
    float or array-like
        The value of the Normal distribution at `x`. If `cumulative` is True,
        returns the CDF value; otherwise, returns the PDF value.

    Examples
    --------
    >>> normdist(0, mean=0, standard_deviation=1, cumulative=True)
    0.8413447460685429

    >>> normdist(0, mean=0, standard_deviation=1, cumulative=False)
    0.3989422804014327

    >>> normdist([0,1,2])
    array([0.5       , 0.84134475, 0.97724987])

    """
    if cumulative:
        return stats.norm.cdf(x, mean, standard_deviation)
    else:
        return stats.norm.pdf(x, mean, standard_deviation)

@hyperlink
def correl(x, y):
    """
    Calculates the Pearson correlation coefficient, or `CORREL() <https://support.google.com/docs/answer/3093990?hl=en&ref_topic=3105600&sjid=5413125246869058878>`_.

    .. math::
       \\frac{\\sum{(x_i - \\bar{x})(y_i - \\bar{y})}}{\\sqrt{\\sum{(x_i - \\bar{x})^2} \\sum{(y_i - \\bar{y})^2}}}

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
    >>> correl([0,1,1], [2,-9,2])
    -0.5

    """
    return np.corrcoef(x,y)[0][1]

@hyperlink
def average(*args):
    """
    Computes the arithmetic mean, or `AVERAGE() <https://support.google.com/docs/answer/3093615?sjid=720707396607486715>`_.

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

@hyperlink
def stdevp(*args):
    """
    Computes the population standard deviation, or `STDEVP() <https://support.google.com/docs/answer/3094105?hl=en&sjid=17542111072464572565>`_.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both stdevp([1,2]) and stdevp(1,2) are valid.

    Returns
    -------
    float
        Population standard deviation of the input data.

    Examples
    --------
    >>> data = [-1, 0, 1]
    >>> stdevp(data)
    0.816496580927726

    >>> stdevp(-1,0,1)
    0.816496580927726

    """
    a = args_to_array(args)
    return np.std(a, ddof=0)

@hyperlink
def varp(*args):
    """
    Computes the population variance, or `VARP() <https://support.google.com/docs/answer/3094113?hl=en&sjid=5413125246869058878>`_.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both varp([1,2]) and varp(1,2) are valid.

    Returns
    -------
    float
        Population variance of the input data.

    Examples
    --------
    >>> data = [-1, 0, 1]
    >>> varp(data)
    0.6666666666666666

    >>> varp(-1,0,1)
    0.6666666666666666

    """
    a = args_to_array(args)
    return np.var(a, ddof=0)

@hyperlink
def stdev(*args):
    """
    Computes the sample standard deviation, or `STDEV() <https://support.google.com/docs/answer/3094054?hl=en&sjid=5413125246869058878>`_.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both stdev([1,2]) and stdev(1,2) are valid.

    Returns
    -------
    float
        Sample standard deviation of the input data.

    Examples
    --------
    >>> data = [-1, 0, 1]
    >>> stdev(data)
    1.0

    >>> stdev(-1,0,1)
    1.0

    """
    a = args_to_array(args)
    return np.std(a, ddof=1)

@hyperlink
def var(*args):
    """
    Computes the sample variance, or `VAR() <https://support.google.com/docs/answer/3094063?hl=en&sjid=5413125246869058878>`_.

    Parameters
    ----------
    args : array_like or numeric scalars
        Input data. This can be a single array-like object or individual numbers.
        Both var([1,2]) and var(1,2) are valid.

    Returns
    -------
    float
        Sample variance of the input data.

    Examples
    --------
    >>> data = [-1, 0, 1]
    >>> var(data)
    1.0

    >>> var(-1,0,1)
    1.0

    """
    a = args_to_array(args)
    return np.var(a, ddof=1)

def change_df_std():
    original_std = pd.DataFrame.std
    def pop_std(self, **kwargs):
        if 'ddof' not in kwargs:
            kwargs['ddof'] = 0
        return original_std(self, **kwargs)
    pd.DataFrame.stdevp = pop_std
    pd.DataFrame.stdev = original_std

def sheets_setup():
    change_df_std()

def change_std_behavior(pd_obj):
    original = getattr(pd_obj, 'std')
    pop_std, sample_std = modify_std(original)
    setattr(pd_obj, 'stdev', sample_std)
    setattr(pd_obj, 'stdevp', pop_std)

def apply_pd_changes():
    change_std_behavior(pd.DataFrame)
    change_std_behavior(pd.Series)

def sheets_setup():
    # silence statsmodels warnings
    warnings.filterwarnings("ignore",
                message="omni_normtest is not valid with less than 8 observations")
    # statsmodels graphics
    warnings.filterwarnings("ignore", 
                message="Series.__getitem__ treating keys as positions is deprecated")
    apply_pd_changes()