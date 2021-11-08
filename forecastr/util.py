#!/usr/bin/env python

import time, os, stat, logging
logger = logging.getLogger(__name__)

from db_queries import get_cause_metadata, get_ids
# from fbd_research.scalars.utils import get_vaccine_reis
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


def compare_arrays(x, y, idx=None, idy=None, mean_draw=True, signif=None, by='year_id'):
    """
    intended to compare forecasting output to mine
    """
    if idx is not None:
        x = x.sel(idx)
    if idy is not None:
        y = y.sel(idy)
    
    by = [by] if by is not list else by

    diff = x.mean('draw') - y.mean('draw') if mean_draw else x - y
    diff_by = diff.mean(list(set(diff.dims) - set(by)))
    if signif is not None:
        diff_by = diff_by.round(signif)
    return diff_by.to_dataframe('diff')['diff']


def year_arg_to_intlist(years, component=None):
    """"
    takes a few different ways you could get a years argument and makes sure a list is returned
    """
    if type(years) is YearRange:
        if component is not None:
            if component == 'past':
                return list(years.past_years)
            elif component == 'future':
                return list(years.forecast_years)
            else:
                return list(years.years)
        else:
            return years.years.tolist()
    elif type(years) is list:
        return years
    elif type(years) is int:
        return [years]
    elif type(years) is str:
        return year_arg_to_intlist(parse_yearstring(years))
    else:
        raise TypeError


def ensure_intlist(arg):
    """
    makes sure that an argument that needs to be an integer list of years is one, handling reasonable alternatives
    """
    if type(arg) is YearRange:
        return year_arg_to_intlist(arg)
    elif type(arg) is int:
        return [arg]
    elif type(arg) is list:
        return arg
    elif type(arg) is tuple or type(arg) is np.ndarray:
        return list(arg)
    else:
        raise TypeError


def year_default(to_list=False, gbd_round_id=5):
    """
    default forecasting year range
    """
    years = YearRange(1990, gbd_round_year_mapping(gbd_round_id), 2040)
    if to_list:
        return year_arg_to_intlist(years)
    else:
        return years


def age_default():
    """
    default set of age_group_ids
    """
    return list(tuple(range(2, 21)) + (30, 31, 32, 235))


def location_default():
    """
    default location set
    """
    return list(db.get_modeled_locations(5)["location_id"].unique())


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


def gbd_round_year_mapping(gbd_round_id):
    """
    maps a gbd round id to the last year of the associated data
    """
    round_year_mapping = {6: 2019, 5: 2017, 4: 2016}
    return round_year_mapping.get(gbd_round_id)
    

def replace_codes_with_labels(data, gbd_round_id=5, inplace=True):
    """
    replace values with labels inplace
    """
    lookup = {
        'location_id': get_ids('location'),
        'age_group_id': get_ids('age_group'),
        'sex_id': get_ids('sex').rename(columns={'sex': 'sex_name'}),
        'rei_id': get_ids('rei'),
        'cause_id': get_ids('cause')
    }

    if type(data) is xr.core.dataarray.DataArray:
        for d in set(data.dims).intersection(set(lookup.keys())):
            data[d].values = [lookup[d].query(f'{d} == {v}').filter(regex='name').values.tolist()[0][0] for v in data[d].values]
    elif type(data) is pd.core.frame.DataFrame:
        for k in lookup.keys():
            if k in data.columns:
                name_prefix = k.split('_')
                name = '_'.join(name_prefix[0:(len(name_prefix) - 1)]) + '_name'
                mapper = pd.Series(lookup[k][name].tolist(), index=lookup[k][k].tolist()).to_dict()
                data[k] = data[k].map(mapper)

    if inplace:
        return None
    else:
        return data


def extract_ids_from_da(da, names):
    """
    extracts ids from DataArrays which are integers and returns an intlist containing them
    """
    ids = []
    for n in name:
        try:
            ids.append(int(da[n].unique()))
        except:
            print(f'{n} not found in DataArray')
            pass
    return ids
        

def get_parents(cause_id, cause_set_id=1, gbd_round_id=5, variables=['cause_id'], return_list=False):
    """
    returns metadata about ancestor causes
    """
    cause_metadata = get_cause_metadata(cause_set_id=cause_set_id, gbd_round_id=gbd_round_id)

    ancestors = []
    parent = cause_id
    while parent != 294:
        parent = cause_metadata[cause_metadata['cause_id'] == parent].parent_id.tolist()[0]
        ancestors.append(parent)

    if return_list:
        return ancestors
    else:
        ret = cause_metadata[cause_metadata['cause_id'].isin(ancestors)]
        return ret[variables]


def get_children(cause_id, cause_set_id=1, gbd_round_id=5, variables=['cause_id'], return_list=False):
    """
    returns metadata about child causes
    """
    cause_metadata = get_cause_metadata(cause_set_id=cause_set_id, gbd_round_id=gbd_round_id)

    children = []
    child = [cause_id]
    while len(child) > 0:
        child = cause_metadata[cause_metadata['parent_id'].isin(child)].cause_id.tolist()
        children.append(child)

    flatten = lambda l: [item for sublist in l for item in sublist]
    children = flatten(children)

    if return_list:
        return children
    else:
        ret = cause_metadata[cause_metadata['cause_id'].isin(children)]
        return ret[variables]


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


def is_restricted(cause_id, sex_id, age_group_id, location_id):
    """
    looks to see whether a given cause/sex/age_group/location is restricted (i.e. not modelled) according to serena's file
    """
    
    if sex_id is not None:
        mapper = {1: 'male', 2: 'female'}
        sex = mapper.get(sex_id)
        sex_based = np.load(f'/ihme/csu/restrictions/{sex}_ages_mortality.npy', allow_pickle=True).item()
        age_restrictions = sex_based.get(cause_id)

        
    if location_id is not None:
        locations = np.load('/ihme/csu/restrictions/location_mortality_restrictions.npy', allow_pickle=True).item()
        location_table = get_ids('location')

        restricted = locations.get(cause_id)
        if restricted is not None:
            location_restrictions = location_table['location_id'][location_table['location_name'].isin(restricted)]
        else:
            location_restrictions = []

    return location_id in location_restrictions and age_group_id in age_restrictions


# def get_vaccine_rei_ids(gbd_round_id=5):
#     """
#     queries the forecasting db (which one) for a list of vaccine reis, then looks up the id for them using centralcomp's shared function
#     """
#     reis = get_vaccine_reis(gbd_round_id)
#     risk_table = get_ids('rei')
#     return risk_table['rei_id'][risk_table['rei'].isin(reis)].tolist()


here = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), x))


def parse_yearstring(arg):
    """
    takes an argument of the form '1990:2017:2040', splits on the colon, and turns it into a fbd_core YearRange object
    """
    bounds = [int(s) for s in arg.split(':')]
    return YearRange(bounds[0], bounds[1], bounds[2])


def spread_dataarray(da, age_group_id=None, location_id=None, sex_id=None):
    """
    da: a dataarray with values that can be spread across sexes, locations, or age groups (e.g. data that pertains to both sexes, all locations, all ages, etc.)
    age_group_id: an integer (list) of age_group_ids to spread values to
    sex_id: an integer (list) of sex_ids to spread values to
    location_id: an integer (list) of location_ids to spread values to

    returns a data array with either the default (see utils.age_group_default and utils.location_default) age_groups and locations or the user specified ones, with
    values spread from the top level group
    """
    
    age_group_id = age_default() if age_group_id is None else age_group_id
    location_id = location_default() if location_id is None else location_id
    sex_id = [1, 2] if sex_id is None else sex_id
    
    if 'sex_id' in da.dims:
        values = da['sex_id'].values.tolist()
        if len(values) == 1 and values[0] == 3:
            da = expand_dimensions(da, da.values, sex_id=sex_id).sel(dict(sex_id=sex_id))

    if 'location_id' in da.dims:
        values = da['location_id'].values.tolist()
        if len(values) == 1 and values[0] == 1:
            da = expand_dimensions(da, da.values, location_id=location_id)

    if 'age_group_id' in da.dims:
        values = da['age_group_id'].values.tolist()
        if len(values) == 1 and values[0] == 22:
            da = expand_dimensions(da, da.values, age_group_id=age_group_id)

    return da
