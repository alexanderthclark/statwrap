'''
Stats functions adapted to the conventions of Freedman, Pisani, and Purves 2007.
'''
import numpy as np
import pandas as pd
from IPython.core.magic import register_line_magic
from statwrap.utils import modify_std

def sd(a):
    """
    Computes the population standard deviation.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.

    Returns
    -------
    float
        The population standard deviation of the input array.

    Examples
    --------
    >>> sd([-1, 0, 1])
    0.816496580927726

    """
    return np.std(a, ddof=0)

def var(a):
    """
    Computes the population variance.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.

    Returns
    -------
    float
        The population variance of the input array.

    Examples
    --------
    >>> var([-1, 0, 1])
    0.6666666666666666

    """
    return np.var(a, ddof=0)

def sd_plus(a):
    """
    Computes the sample standard deviation.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.

    Returns
    -------
    float
        The sample standard deviation of the input array.

    Examples
    --------
    >>> sd_plus([-1, 0, 1])
    1.0

    """
    return np.std(a, ddof=1)

def var_plus(a):
    """
    Computes the sample variance.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.

    Returns
    -------
    float
        The sample variance of the input array.

    Examples
    --------
    >>> var_plus([-1, 0, 1])
    1.0

    """
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

def fpp_setup():
	apply_pd_changes()
