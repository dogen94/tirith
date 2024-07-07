
# Standard Library
import os
from functools import wraps


def ensure_cwd(*a, **kw):
    def wrap(func):
        @wraps(func)
        def func_wrapper(*a, **kw):
            # Store cwd
            cwd = os.getcwd()
            # Get desired dir
            ddir = kw.pop("ddir", None)
            # Go to ddir
            if ddir:
                os.chdir(ddir)
            # Run func
            fout = func(*a, **kw)
            # Return back to cwd
            if ddir:
                os.chdir(cwd)
            return fout
        return func_wrapper
    return wrap