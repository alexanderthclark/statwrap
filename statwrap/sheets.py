'''
Stats functions adapted to the conventions of Google Sheets.
'''
import numpy as np
import pandas as pd
from IPython.core.magic import register_line_magic
from statwrap.utils import modify_std

def stdevp(a):
	'''
	Computes the population standard deviation, or STDEVP().
	'''
	return np.std(a, ddof=0)

def varp(a):
	'''
	Computes the population variance, or VARP().
	'''
	return np.var(a, ddof=0)

def stdev(a):
	'''
	Computes the sample standard deviation, or STDEV().
	'''
	return np.std(a, ddof=1)

def var(a):
	'''
	Computes the sample variance, or VAR().
	'''
	return np.var(a, ddof=1)

def stdevp(a):
	"""
	Computes the population standard deviation, or STDEVP().

	Parameters
	----------
	a : array_like
	    Input data.

	Returns
	-------
	float
	    Population standard deviation of the input data.

	Examples
	--------
	>>> data = [-1, 0, 1]
	>>> stdevp(data)
	0.816496580927726

	"""
	return np.std(a, ddof=0)

def varp(a):
	"""
	Computes the population variance, or VARP().

	Parameters
	----------
	a : array_like
	    Input data.

	Returns
	-------
	float
	    Population variance of the input data.

	Examples
	--------
	>>> data = [-1, 0, 1]
	>>> varp(data)
	0.6666666666666666

	"""
	return np.var(a, ddof=0)

def stdev(a):
	"""
	Computes the sample standard deviation, or STDEV().

	Parameters
	----------
	a : array_like
	    Input data.

	Returns
	-------
	float
	    Sample standard deviation of the input data.

	Examples
	--------
	>>> data = [-1, 0, 1]
	>>> stdev(data)
	1.0

	"""
	return np.std(a, ddof=1)

def var(a):
	"""
	Computes the sample variance, or VAR().

	Parameters
	----------
	a : array_like
	    Input data.

	Returns
	-------
	float
	    Sample variance of the input data.

	Examples
	--------
	>>> data = [-1, 0, 1]
	>>> var(data)
	1.0

	"""
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