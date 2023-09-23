'''
Stats functions adapted to the conventions of Google Sheets.
'''
import numpy as np
import pandas as pd
from IPython.core.magic import register_line_magic
from statwrap.utils import modify_std, args_to_array

def average(*args):
	"""
    Computes the arithmetic mean, or `AVERAGE() <https://support.google.com/docs/answer/3093615?sjid=720707396607486715>`.

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

def stdevp(*args):
	"""
	Computes the population standard deviation, or `STDEVP() <https://support.google.com/docs/answer/3094105?hl=en&sjid=17542111072464572565>`.

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

def varp(*args):
	"""
	Computes the population variance, or VARP().

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

def stdev(*args):
	"""
	Computes the sample standard deviation, or STDEV().

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

def var(*args):
	"""
	Computes the sample variance, or VAR().

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
	apply_pd_changes()