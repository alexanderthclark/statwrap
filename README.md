
# StatWrap

This is a package meant for intro statistics students who are also new to Python, or at least its statistical libraries.

Different submodules adopt different conventions. The `fpp` submodule corresponds to *Statistics* by Freedman, Pisani, and Purves while the `sheets` module corresponds to Google Sheets conventions.

# IPython Usage

The target user will use this package in a Google Colab or Jupyter notebook. Install with `!pip install statwrap`.

Then import the package and use a magic command to load the desired module. 
```python
import statwrap
%use_fpp
```
`%use_fpp` imports the functions that adhere to the conventions of Freedman, Pisani, and Purves. This will also overwrite and introduce new pandas methods for working with DataFrames and Series. `%use_sheets` is similar, but it borrows the conventions of Google Sheets. `%use_all` loads in both fpp and sheets conventions as well as user experience functions such as a data upload widget.

# Design and Style 

Python is known for its simplicity and readability. Still, someone new to statistics and programming might find themselves intimidated by the many imports necessary to work with data and run statistical tests. A user will find conflicting defaults across different packages and terminology can differ. This package aims to reduce the mental overhead required to juggle the notation they find in a textbook with formula names in Google Sheets and what they find in packages like NumPy, pandas, scipy, and statsmodels. 

The design principles of this package are mostly the design principles of Python. However, we don't always adhere to the principle that "explicit is better than implicit" and we prefer convention to configuration, with the configuration being done once with a magic command like `%use_fpp`. 

# Bugs and Feature Requests

If you find a bug or think of a useful feature, please open an issue here on Github, please see our [Contributors Guide](.github/CONTRIBUTING.md) for more details.

# Contribute

If you are interested in using this in your classroom or contributing, please reach out to me at alexander.clark@columbia.edu. For more detailed guidelines on how to contribute, please refer to our [Contributors Guide](.github/CONTRIBUTING.md).

This project is sponsored by [Zulip](https://zulip.com).  
<img src="https://raw.githubusercontent.com/zulip/zulip/32d5b4fe3eae40c3cefcdc29c83074827f6a903f/static/images/logo/zulip-icon-circle.svg" width="30">
