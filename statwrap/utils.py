'''
These are utilities agnostic to specific conventions.
'''
import numpy as np
import functools

class Formula:
    '''
    This class is used to modify the display behavior of functions that have a mathematical formula.
    '''
    def __init__(self, func):
        '''

        Parameters
        ----------
        func : function
            Function to be modified, with a formula inside a math block in the docstring.
        '''
        self.func = func
        functools.update_wrapper(self, func)
        self.__doc__ = func.__doc__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.func, name)

    def _repr_latex_(self):
        lines = self.func.__doc__.splitlines()
        start = False

        # this supports only one-line formulas
        for line in lines:
            if '.. math::' in line:
                start = True
            elif start:# and not line.strip(): #and not line.startswith(' '):
                latex = line.strip()
                break

        return f"$$ {latex} $$"

def formula(func):
    """
    Decorator to modify the display behavior of functions with a mathematical formula.
    The function should have its formula inside a math block in its docstring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    def _repr_latex_(self):
        lines = func.__doc__.splitlines()
        start = False

        # this supports only one-line formulas
        for line in lines:
            if '.. math::' in line:
                start = True
            elif start:
                latex = line.strip()
                break

        return f"$$ {latex} $$"
    
    setattr(wrapper, "_repr_latex_", _repr_latex_)

    return wrapper

def args_to_array(args):
    """
    When args is a tuple of scalars, this returns them in one array.
    When args' first element is iterable, this returns the first element.
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