#!/usr/bin/env python

import time, os, stat, logging
logger = logging.getLogger(__name__)

from scipy.special import expit
import xarray as xr
import pandas as pd
import numpy as np

xr.set_options(file_cache_maxsize=1000)


def file_age_in_seconds(pathname):
    try:
        age = os.stat(pathname)[stat.ST_MTIME]
    except:
        age = 0.
    return time.time() - age


def is_stale(filename, days=30):
    """
    checks if a file is old and returns a boolean. if you pass in the override flag (a keyword argument)
    then the function will return True (the file is stale)
    """
    stale = days * 60 * 24 * 60
    age = file_age_in_seconds(filename)
    return age > stale


def inverse_transform(da, transform="logit", **kwargs):
    """
    arcing is done in logit or log space. after it is done the inverse function is applied
    """
    if transform == "logit":
        da = expit(da)
    elif transform == "log":
        da = np.exp(da)
    else:
        pass
    return da


def root_mean_square_error(predicted, observed, skipna=True):
    return np.sqrt(((predicted - observed) ** 2).mean(skipna=skipna))


def flatten(l, unique=False):
    """
    takes a list of lists and returns a list which doesn't contain any lists, optionally removes duplicated values using sets
    """
    ret = [item for sublist in l for item in sublist]
    if unique:
        return list(set(ret))
    else:
        return ret

    
def coords_of_missing(da):
    """
    takes a data array and returns a pandas dataframe of the coordinates of values which are nan
    """
    res = da.where(np.isnan(da), drop=True)
    return to_df(res)


def to_df(da, name='default'):
    """
    converts an xarray dataarray or dataset to a pandas dataframe and tries to make the naming sensible
    """
    
    if type(da) is xr.core.dataset.Dataset:
        da = da.rename({list(da.variables)[-1]: name})
        df = da.to_dataframe()
    else:
        df = da.to_dataframe(name if da.name is None else da.name)
    return df


def unique_coords(df, along):
    """
    takes a pandas dataframe, finds the unique values of each column (including the index) and returns a list of lists of the unique values
    """
    df.reset_index(inplace=True)
    return [df[d].unique().tolist() for d in df[along]]


here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))
