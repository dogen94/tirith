
# Standard Library
import os
import json
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


def read_json(fjson):
    if os.path.isfile(fjson):
        with open(fjson) as fp:
            # Read what we just got and extract url
            jout = json.load(fp)
        return jout
    else:
        raise FileExistsError(f"{fjson} does not exist")