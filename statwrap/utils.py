'''
These are utilities agnostic to specific conventions.
'''
import numpy as np
import functools
import re
import pandas as pd
import statwrap.fpp as fpp
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import (
    plot_partregress_grid,
    plot_fit
)

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
    
    def _repr_latex_():
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

def find_first_external_link(s):
    """
    Find the first external link in a string formatted as `LinkText <URL>`.

    Parameters:
    s (str): The string to search in.

    Returns:
    (str, str): The first external link text and URL found, or (None, None) if no link is found.
    """
    link_pattern = r'`([^`]+) <(https?://[^\s>]+)>`_'
    match = re.search(link_pattern, s)
    return match.groups() if match else (None, None)

def hyperlink(func):
    """
    Decorator to modify the display behavior of functions with a hyperlink.
    The function should have its hyperlink inside its docstring.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    def _repr_html_():
        link_text, hyperlink = find_first_external_link(func.__doc__)
        if hyperlink:
            return f"<a href='{hyperlink}' target='_blank'>{link_text}</a>"
        else:
            return ""
    
    setattr(wrapper, "_repr_html_", _repr_html_)
    
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


class Hyperplane:
    """
    Represents a hyperplane in a multidimensional space.

    The hyperplane is represented by the equation form:

    .. math::

        y = c_0 + c_1 x_1 + c_2 x_2 + \ldots + c_n x_n

    where `c_i` are the coefficients and `x_i` are the independent variables.

    Parameters
    ----------
    *coefficients : float
        The coefficients defining the hyperplane. `c_0` is the constant term,
        and `c_1, c_2, ..., c_n` are the coefficients of the variables `x_1, x_2, ..., x_n`.

    Attributes
    ----------
    coefficients : ndarray
        An array holding the coefficients of the hyperplane.

    Examples
    --------
    >>> plane = Hyperplane(1, 1, 1)
    >>> plane(0, 1)
    2

    """

    def __init__(self, *coefficients):
        """
        Initializes a new instance of the Hyperplane class.

        Parameters
        ----------
        *coefficients : float
            The coefficients defining the hyperplane.
        """
        array = args_to_array(coefficients)
        self.coefficients = np.squeeze(np.array(coefficients))

    def __call__(self, *args):
        """
        Computes the value of the hyperplane for the given variables.

        Parameters
        ----------
        *args : float
            The values of the variables `x_1, x_2, ..., x_n`.

        Returns
        -------
        float
            The value of the hyperplane for the given variables.

        Examples
        --------
        >>> plane = Hyperplane(1, 1, 1)
        >>> plane(0, 1)
        2

        """
        return self.coefficients[0] + np.dot(self.coefficients[1:], args)

    def _repr_latex_(self):
        """
        Returns a LaTeX representation of the hyperplane.

        This method is used by IPython for rendering the hyperplane in a
        Jupyter Notebook.

        Returns
        -------
        str
            A LaTeX string representing the hyperplane.

        Examples
        --------
        >>> plane = Hyperplane(1, 2, 4)
        >>> plane._repr_latex_()
        '$\\hat{y} = 1 + 2 x_1 + 4 x_2$'

        """
        terms = [f'{round(self.coefficients[0],3):g}']
        for i, coef in enumerate(self.coefficients[1:], 1):
            coef = round(coef, 3)
            terms.append(f'{coef:+g} x_{{{i}}}')
        return r'$\hat{y} = ' + ' '.join(terms) + "$"

    def predict(self, data, add_constant = True, dataframe = True):

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if add_constant:
            data.insert(0, 'constant', 1)

        return data.to_numpy() @ self.coefficients


class RegressionLine(Hyperplane):
    """
    RegressionLine class extends Hyperplane to model a univariate regression line
    with given coefficients, input values (x), and target values (y).

    Attributes
    ----------
    y : array-like
        Target values.
    x : array-like
        Input values.
    coefficients : tuple
        Coefficients for the hyperplane.
    residuals : array-like
        Residuals of the regression.
    predictions : array-like
        Predicted values based on input x.
    rms_error : float
        Root Mean Square Error of the regression.
    """

    def __init__(self, y, x, results):
        """
        Initializes a RegressionLine instance.

        Parameters
        ----------
        y : array-like
            Target values.
        x : array-like
            Input values.
        coefficients : tuple
            Coefficients for the hyperplane.
        """
        super().__init__(*results.params)
        self.__results = results
        self.__y = y
        self.__x = x
        self.__predictions = self.predict(self.__x)
        self.__residuals = self.__results.resid.round(5)
        self.__rms_error = np.sqrt(np.mean(self.__residuals**2))

    @property
    def y(self):
        """Returns the target values."""
        return self.__y

    @property
    def x(self):
        """Returns the input values."""
        return self.__x
   
    @property
    def results(self):
        """Returns the StatsModels results object."""
        return self.__results
   
    @property
    def predictions(self):
        """Returns the predicted values based on input x."""
        return self.__predictions

    @property
    def residuals(self):
        """Returns the residuals of the regression."""
        return self.__residuals

    @property
    def rms_error(self):
        """Returns the Root Mean Square Error of the regression."""
        return self.__rms_error

    def partial_regression_plot(self, show = True):
        """Shows a partial regression plot for each predictor variable."""
        f = plot_partregress_grid(self.results)
        if show:
            plt.show()
        else:
            return f

    def scatter_plot(self, **kwargs):
        """Shows a scatter plot for the data."""
        if False: #len(self.results.params) == 2:
            if 'regression_line' not in kwargs:
                kwargs['regression_line'] = True
            return fpp.scatter_plot(self.x, self.y, **kwargs)
        else:
            tmp = pd.DataFrame(self.x)
            ncol = len(tmp.columns)
            fig, axs = plt.subplots(1, ncol, sharey=True, squeeze=False)
            for key, col in enumerate(tmp.columns):
                x0 = tmp[col]
                ax = axs[0, key]
                fpp.scatter_plot(x0, self.y, ax=ax, show=False)
            plt.show()

    def residual_plot(self, **kwargs):
        """Shows a scatter plot of x vs the residuals."""
        y = self.residuals
        if False: #len(self.results.params) == 2:
            if 'regression_line' not in kwargs:
                kwargs['regression_line'] = True
            return fpp.scatter_plot(self.x, y, **kwargs)
        else:
            tmp = pd.DataFrame(self.x)
            ncol = len(tmp.columns)
            fig, axs = plt.subplots(1, ncol, sharey=True, squeeze=False)
            for key, col in enumerate(tmp.columns):
                x0 = tmp[col]
                ax = axs[0, key]
                fpp.scatter_plot(x0, y, ax=ax, show=False)
                ax.axhline(0, color = 'black', lw = 0.5)
            plt.show()