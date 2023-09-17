'''
Main setup utilities.
'''
from IPython import get_ipython
from IPython.core.magic import register_line_magic
from statwrap.fpp import fpp_setup
from statwrap.sheets import sheets_setup

@register_line_magic
def use_fpp(line):
	'''
	Load the fpp module. 

	This imports functions that adhere to the conventions found in "Statistics" by Freedman, Pisani, and Purves.


	Parameters
	----------
	line : str
		Unused parameter retained for compatibility with IPython line magic.

	Returns
	-------
	None

	Examples
	--------
	Using this function in IPython:

	.. code-block:: python

		%use_fpp
	'''
	ip = get_ipython()
	ip.ex("from statwrap.fpp import *")
	fpp_setup()

@register_line_magic
def use_sheets(line):
	'''
	Load the sheets module. 

	This imports functions that adhere to the conventions specific to Google Sheets.

	Parameters
	----------
	line : str
		Unused parameter retained for compatibility with IPython line magic.

	Returns
	-------
	None

	Examples
	--------
	Using this function in IPython:

	.. code-block:: python

		%use_sheets
	'''
	ip = get_ipython()
	ip.ex("from statwrap.sheets import *")
	sheets_setup()
