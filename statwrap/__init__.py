try:
    __IPYTHON__
    from .core import use_fpp
    from .core import use_sheets
except NameError:
    pass
