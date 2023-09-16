'''
Main setup utilities.
'''
from IPython import get_ipython
from IPython.core.magic import register_line_magic
from statwrap.fpp import fpp_setup
from statwrap.sheets import sheets_setup

@register_line_magic
def use_fpp(line):
	ip = get_ipython()
	ip.ex("from statwrap.fpp import *")
	fpp_setup()

@register_line_magic
def use_sheets(line):
	ip = get_ipython()
	ip.ex("from statwrap.sheets import *")
	sheets_setup()

def modify_std(original_method):
	"""
	Modifies a standard deviation method to adjust the 'ddof' parameter.

	Parameters
	----------
	original_method : callable
		The original method for standard deviation or variance that accepts a 'ddof' parameter.

	Returns
	-------
	tuple of callables
		A tuple containing two modified methods:
		- `pop_std` for population standard deviation (ddof=0)
		- `sample_std` for sample standard deviation (ddof=1).

	Notes
	-----
	If the 'ddof' parameter is already provided when calling the returned methods,
	it will not be overwritten.
	"""
	def pop_std(self, **kwargs):
		if 'ddof' not in kwargs:
			kwargs['ddof'] = 0
		return original_method(self, **kwargs)
	def sample_std(self, **kwargs):
		if 'ddof' not in kwargs:
			kwargs['ddof'] = 1
		return original_method(self, **kwargs)
	return pop_std, sample_std