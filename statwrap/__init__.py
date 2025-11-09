try:
    __IPYTHON__
    from .core import use_fpp
    from .core import use_sheets
    from .core import use_all
except NameError:
    pass

from .exceptions import StatwrapError, SimplePlotError

__version__ = '0.3.1'

__all__ = ["use_fpp", "use_sheets", "use_all", "__version__", "StatwrapError", "SimplePlotError"]

