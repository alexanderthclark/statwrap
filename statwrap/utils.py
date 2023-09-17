'''
These are utilities agnostic to specific conventions.
'''
import numpy as np

def args_to_array(args):
	"""
	When args is a tuple of scalars, this returns them in one array.
	When args' first element is iterablem this returns the first element.
	"""
	# check for iterable
	if np.iterable(args[0]) and len(args) == 1:
		a = args[0]
	elif np.iterable(args[0]):
		raise ValueError("Unexpected input. args should be a tuple of scalars or a single array")
	else:
		a = tuple(args)
	return a

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