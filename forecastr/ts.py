#!/usr/bin/env python

import logging, click, itertools, os, tempfile, shutil
logger = logging.getLogger(__name__)

import xarray as xr
import numpy as np
import pandas as pd
from random import randint
from scipy.special import logit, expit
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import DataFrame
from rpy2.robjects import pandas2ri, numpy2ri
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.special import logit, expit
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

xr.set_options(file_cache_maxsize=1000)


def forecast_univariate(da, years, name, variables=[], xreg=None, xreg_code=None, substitute_past=False, method='ets',
                        damped=False, power_transform=None, smooth=False, smooth_frac=1./3,
                        transform=None, empirical_var=False, bootstrap=False, simulate=False, tempdir=None, level=None,
                        draws=100, holdout=None, opt_crit='lik', save_params=False,
                        scenarios=False, scenario_quantiles=[.15, .85], params=None,
                        fix_na_sims=False, max_tries=5, scale=False, ts_proportion=.25,
                        logit_upper='max', max_factor=2.,logit_lower=0., strictly_positive=False,
                        offset_zeroes=False, interp_nan=True, noise=None, noise_mode=None,  **kwargs):
    """
    makes time series forecasts for time series grouped by variables (e.g., age group, location, sex)
    """

    # modify the years object if a holdout is desired, then proceed as normal
    # TODO needs to be rewritten to not use yearrange
    if holdout is not None:
        years = YearRange(years.past_start, years.forecast_start - holdout, years.forecast_end)

    # make sure the dataframe only contains past years
    df = df.query(f'year_id < {years.forecast_start}')

    # get index types for decoding
    variable_types = df[variables].dtypes.to_dict()
    # encode the index values into a column name and unstack the dataframe
    pad_size = find_pad_size(df[variables])

    # TODO needs to be rewritten to use generic time index
    df = to_flat_hierarchical_df(df, name, index=['year_id'], variables=variables, pad_size=pad_size)

    # look for time-series with constant variance, remove them for now
    constants = df.var(skipna=True).isin([0., np.NaN, -np.inf, np.inf])
    consecutive = ((df.diff(1) != 0).astype('int').cumsum()).max()
    constants = (constants) | ((consecutive / df.shape[0]) < ts_proportion)
    
    # df.loc[:, (df != df.iloc[0]).any()]
    constant_df = df.loc[:, constants]
    df = df.loc[:, ~constants]

    # if smooth:
    #     df = df.apply(lambda x: lowess(np.asarray(x), df.index, frac=smooth_frac, missing='none')[:,1])

    if offset_zeroes is not None:
        df[df == 0.0] = offset_zeroes

    # turn dataframe into R dataframe, compute the forecast horizon, initialize rpy2
    forecast = importr('forecast')
    pandas2ri.activate()
    rdf = pandas2ri.py2rpy(df)
    rdf.colnames = df.columns.tolist()

    forecast_years = list(set(years.years).difference(set(df.index.values)))
    past_years = list(df.index.values)
    h = len(forecast_years)

    # set method specific default arguments, assign fit/predict/simulate methods
    if method == "arima":
        fit_args = {'stationary': damped, 'seasonal': False}
        fitter = forecast.auto_arima
        simulator = forecast.simulate_Arima
        predictor = forecast.forecast_Arima
    elif method == "ets":
        fit_args = {'damped': damped, 'model': 'ZZN', 'restrict': True, 'bounds': 'admissible', 'opt_crit': opt_crit}
        fitter = forecast.ets
        simulator = forecast.simulate_ets
        predictor = forecast.forecast_ets
    elif method == 'nnetar':
        fit_args = {'P': 0, 'scale_inputs': True}
        fitter = forecast.nnetar
        simulator = forecast.simulate_nnetar
        predictor = forecast.forecast_nnetar
    else:
        raise ValueError

    # pass through anynomous keyword arguments to fit functions
    fit_args.update(**kwargs)
    predict_args = {'bootstrap': bootstrap}

    # pass through box-cox transform arg
    if power_transform is not None:
        fit_args.update({'lambda': power_transform})
        predict_args.update({'lambda': power_transform})

    # set uncertainty parameters (or not) for predict/simulate functions
    if simulate:
        predict_args.update({'future': True, 'nsim': h})
    elif level is not None:
        predict_args.update({'level': level})
    else:
        pass

    # initialization of xreg arg for use with arima
    # TODO make time index generic
    if xreg is not None:
        numpy2ri.activate()
        xreg = xreg.to_dataframe().unstack(xreg_code).reset_index().set_index(variables + ['year_id']).sort_index()
        coef = []

    # TODO make time index generic
    concat_dim = variables + ['year_id']

    # new plan
    # do i need to loop over rdf? why not just df?
    # fastest option is to loop over values directly using numpy apparently
    
    
    # within the loop
    # preprocess (scale, transform, smooth)
    #  - create or re-use a transformer class which contains relevant parameters

    # define model
    #  - model class which holds model parameters, controls output
    # fit model
    # posprocess (fix nas, invert transform, rescale, save params)
    #  - handle within respective earlier modules

    # loop over individual time-series
    ret = []
    for i, ts in enumerate(rdf):
        # decode column name to assign to correct output location
        code = decoder(rdf.colnames[i], pad_size, variables)
        if smooth:
            ts = lowess(ts, np.arange(0, ts.shape[0]), frac=smooth_frac, missing='none')[:, 1]
        
        if scale:
            mm = MinMaxScaler()
            ts = mm.fit_transform(ts.reshape(-1, 1))[:,0]
            invert_scale = lambda x: pd.Series(mm.inverse_transform(np.asarray(x).reshape(x.shape[0], 1))[:,0])
        
        if transform == 'log':
            ts = np.log(ts)
            invert = lambda x: np.exp(x)
        elif transform in ['yeo-johnson', 'box-cox']:
            pt = PowerTransformer()
            ts = pt.fit_transform(ts.reshape(-1, 1))[:,0]
            invert = lambda x: pd.Series(pt.inverse_transform(np.asarray(x).reshape(x.shape[0], 1))[:,0])
        elif transform == 'logit':
            ts = logit(ts)
            invert = lambda x: expit(x)
        elif transform == 'scaled-logit':
            if isinstance(logit_upper, str):
                if logit_upper == 'max':
                    upper_cap = ts.max() * max_factor
                elif logit_upper == 'last':
                    upper_cap = ts[len(ts) - 1] * max_factor
            elif isinstance(logit_upper, pd.Series):
                upper_cap = filter_by_code(logit_upper, variables, code, variable_types)
            else:
                upper_cap = logit_upper
            if isinstance(logit_lower, str):
                if logit_lower == 'min':
                    lower_cap = ts.min() / max_factor
                elif logit_lower == "last":
                    lower_cap = ts[len(ts) - 1] / max_factor
            elif isinstance(logit_lower, pd.DataFrame):
                lower_cap = filter_by_code(logit_lower, variables, code, variable_types, len(ts))
            else:
                lower_cap = logit_lower
            # this allows to pass a bound which has length len(ts) + h which is applied to the past here
            # and then passed to the inversion function in its full length
            upper = upper_cap
            lower = lower_cap
            if (isinstance(upper, np.ndarray)) and (len(upper_cap) > len(ts)):
                upper = upper[:len(ts)]
            else:
                upper = upper_cap
            if (isinstance(lower, np.ndarray)) and (len(lower) > len(ts)):
                lower = lower[:len(ts)]
            else:
                lower = lower_cap
            ts = np.log((ts - lower) / (upper - ts))
            invert = lambda x, upper, lower: (upper - lower) * np.exp(x) / (1 + np.exp(x)) + lower
        elif transform == 'log1p':
            ts = np.log1p(ts)
            invert = lambda x: np.expm1(x)
        elif transform == 'arctan':
            ts = np.arctan(ts)
            invert = lambda x: np.tan(x)
        else:
            pass

        if interp_nan:
            nans, miss = nan_helper(ts)
            ts[nans] = miss(ts[nans])
        
        # fit model
        if xreg is None:
            fit = fitter(ts, **fit_args)
        elif params is not None:
            # fit model with fixed parameters for scenarios
            params = params.set_index(variables)
            code_pars = params.loc[tuple(code)].to_dict()
            for p in [par for par in code_pars.keys() if par in ['alpha', 'beta', 'gamma', 'phi']]:
                fit_args.update({p: code_pars.get(p)})
            fit = fitter(ts, **fit_args)
        else:
            # fit model with xreg option if arima and xreg options passed
            try:
                tmp = xreg.loc[tuple(code)]
                rxmat = numpy2ri.py2ri(np.array(tmp.loc[slice(past_years[0], past_years[-1])]))
                rxmat_future = numpy2ri.py2ri(np.array(tmp.loc[slice(forecast_years[0], forecast_years[-1])]))
                fit_args.update({'xreg': rxmat})
                predict_args.update({'xreg': rxmat_future})
                fit = fitter(ts, **fit_args)
                coef.append(numpy2ri.rpy2py(fit.rx2['coef'])[:xreg.shape[1]])
            except:
                fit = fitter(ts, **fit_args)

                
        # TODO fix all this, should output pars
        # extract model parameters
        # if method == 'ets':
        #     pars = fit.rx2('par')
        #     names = ['alpha', 'beta', 'phi', 'l', 'b']
        # elif method == 'arima':
        #     pars = fit.rx2('coef')
        # else:
        #     pars = None
        pars = None

        if pars is not None:
            pars = {pars.names[i]: v for i, v in enumerate(pars)}
            par_names = list(pars.keys())

        if empirical_var:
            predict_args.update({'innov': np.random.normal(0, ts.var(), h)})

        # make predictions
        if simulate:
            preds = pd.DataFrame({d: np.concatenate([fit.rx2('fitted'), simulator(fit, **predict_args)])
                                  for d in range(0, draws)})

            if noise is not None and isinstance(noise, pd.Series) and noise_mode == 'pre-transform':
                preds = preds + filter_by_code(noise, variables, code, variable_types)[:,1].T

            if transform is not None:
                if transform == 'scaled-logit':
                    preds = preds.apply(invert, upper=upper_cap, lower=lower_cap, axis=0)
                    preds.loc[(preds.apply(min, axis=0) < lower_cap) & (preds.apply(max, axis=0) > upper_cap)] = np.NaN
                else:
                    preds = preds.apply(invert, axis=0)

            if noise is not None and isinstance(noise, pd.Series) and noise_mode == 'post-transform':
                preds = preds + filter_by_code(noise, variables, code, variable_types)[:,1].T
                    
            if fix_na_sims:
                tries = 0
                fixed = False
                while (not fixed) & (tries < max_tries):
                    nan_draws = pd.isnull(preds).any(0)
                    if sum(nan_draws) > 0:
                        fixed = False
                    else:
                        fixed = True
                        break
                    
                    nan_draws = nan_draws.loc[nan_draws == True].index
                    for d in nan_draws:
                        preds.iloc[:,d] = np.concatenate([fit.rx2('fitted'), simulator(fit, **predict_args)])
                        if transform is not None:
                            if transform == 'scaled-logit':
                                preds.iloc[:,d] = invert(preds.iloc[:,d], upper=upper_cap, lower=lower_cap)
                            else:
                                preds.iloc[:,d] = invert(preds.iloc[:,d])
                    tries += 1

            if scale:
                preds = preds.apply(invert_scale, axis=0)
            # TODO fix time index
            preds['year_id'] = years.years
        else:
            preds = predictor(fit, h=h, **predict_args)
            if level is not None:
                # TODO time index
                preds = pd.DataFrame({name: np.concatenate([fit.rx2('fitted'), preds.rx2('mean')]),
                                      'lower': np.concatenate([fit.rx2('fitted'), preds.rx2('lower')]),
                                      'upper': np.concatenate([fit.rx2('fitted'), preds.rx2('upper')]),
                                      'year_id': years.years})
                
                if transform is not None:
                    preds[name] = invert(preds[[name]])
                    preds['lower'] = invert(preds[['lower']])
                    preds['upper'] = invert(preds[['upper']])

                if scale:
                    preds[name] = invert_scale(preds[[name]])
                    preds['lower'] = invert_scale(preds[['lower']])
                    preds['upper'] = invert_scale(preds[['upper']])
                    
            else:
                # TODO time index
                preds = pd.DataFrame({name: np.concatenate([fit.rx2('fitted'), preds.rx2('mean')]),
                                      'year_id': years.years})
                if transform is not None:
                    preds[name] = invert(preds[[name]])
        
        # assign decoded index values as columns
        for j, c in enumerate(code):
            preds[variables[j]] = c

        # save model parameters in output
        if save_params or scenarios:
            for k in pars.keys():
                preds[k] = pars.get(k)

        if tempdir is not None:
            preds.to_csv(f'{tempdir}/{"_".join([str(c) for c in code])}.csv', index=False)
        else:
            ret.append(preds)

    if tempdir is None:
        ret = pd.concat(ret)
    else:
        ret = pd.concat([pd.read_csv(f) for f in os.listdir(f'{tempdir}/*.csv')])

    ret[variables] = ret[variables].astype(variable_types)

    # wide-long draw reformatting
    if simulate:
        if save_params or scenarios:
            ret = pd.melt(ret, id_vars=concat_dim + par_names, value_name=name, var_name='draw')
        else:
            ret = pd.melt(ret, id_vars=concat_dim, value_name=name, var_name='draw')

        # ret['draw'] = ret['draw'].apply(lambda x: x.replace('draw_', '')).astype(int)
        concat_dim.append('draw')

    ret = ret.set_index(concat_dim)

    # fix non-slope parameters, take quantiles of slope parameters across draws,
    # then recursively call this function with the fixed parameters, attach output to
    # the reference scenario
    if scenarios:
        if method == 'ets':
            par_names = [p for p in par_names if p in ['alpha', 'beta', 'phi', 'gamma']]
            params = ret.groupby(variables).mean()[par_names].reset_index()
            # TODO remove location id reference here
            grouped_params = params.groupby([v for v in variables if v != 'location_id'] + \
                                            [p for p in par_names if p != 'beta'])['beta']
            # take quantiles across locations for the slope param
            scenario_params = []
            for q in scenario_quantiles:
                betas = grouped_params.quantile(q).reset_index()
                betas['quantile'] = q
                scenario_params.append(betas)
            scenario_params = pd.concat(scenario_params)
            params = pd.merge(scenario_params, params.drop('beta', axis=1))

            # compute scenario predictions
            ret_scenarios = []
            for q in scenario_quantiles:
                scenario = forecast_univariate(da=da_mean, years=years, name=name, variables=variables, damped=damped,
                                               power_transform=power_transform, simulate=simulate, draws=draws, level=level,
                                               bootstrap=bootstrap, scenarios=False,
                                               params=params.loc[params['quantile'] == q].drop('quantile', axis=1),
                                               save_params=save_params)
                ret_scenarios.append(scenario.assign_coords({'scenario': q}))
                
            ret_scenarios = xr.concat(ret_scenarios, dim='scenario')

    if strictly_positive:
        ret.loc[ret[name] < 0.] = 0.

    ret = ret.to_xarray()

    if scenarios:
        if not save_params:
            ret = ret.drop_vars(par_names)
            # ret = ret[name]
        ret = xr.concat([ret.assign_coords({'scenario': 'reference'}), ret_scenarios], dim='scenario')

    # what it says
    if substitute_past:
        future = ret.sel(dict(year_id=years.forecast_years.tolist()))
        da = da.sel(dict(year_id=years.past_years.tolist()))
        da, future = xr.broadcast(da, future[name])
        ret = xr.concat([da, future], dim='year_id')
    else:
        ret = ret

    # if any constant time-series were found, forward-fill their values and merge them back in
    if not constant_df.empty:
        constant_df = constant_df.fillna(method='ffill')
        constant_df = from_flat_hierarchical_df(constant_df, name, ['year_id'], variables, pad_size)
        constants = constant_df.to_xarray()
        ret = xr.merge([ret, constants])
        
    return ret



def to_flat_hierarchical_df(df, name, to_drop=[], index='year_id', variables=[], pad_size=None, padder='0'):
    """
    takes a hierarchically indexed DataFrame (i.e., a DataArray turned into a DataFrame), and then flattens the index into a string code
    which is just the integer value of the index padded to be the same length as the longest index value across any of the indices

    e.g., it turns this DataFrame


    sex_id, location_id, sev
    1       1            .01
    ...     ...          ...
    2       10           .02
    
    into this DataFrame

    0101 0210

    .01  .02

    this is intended as input to Hyndman's gts/hts functions in R

    """
    df = df.pivot_table(index=index, columns=variables, values=name)
    if type(variables) is list and len(variables) > 1:
        new_cols = [''.join([str(s).rjust(pad_size[i], padder) for i, s in enumerate(col)]) for col in df.columns]
        df.columns = new_cols
    else:
        df.columns = df.columns.astype(str)
        df.columns = [col.rjust(pad_size, padder) for col in df.columns]

    return df


def decoder(s, pad_size, variables):
    """
    takes a string, slices it using encoding pad size and returns dict
    """
    cursor = 0
    res = []
    slen = len(s)

    for i, v in enumerate(variables):
        end = pad_size[i] + cursor
        end = slen if end <= cursor else end
        res.append(s[cursor:end])
        cursor += pad_size[i]

    return res
    
    
def from_flat_hierarchical_df(df, name, index=['year_id'], variables=[], pad_size=None):
    """
    take the flat hierarchical index encoding created by to_flat_hierarchical_df and reverse it
    """
    indices_to_add = [idx for idx in index if idx not in df.index.names]
    if len(indices_to_add) > 0:
        df = df.set_index(indices_to_add, append=True)
    ts = df.stack().rename(name)
    names = index + ['code']
    df = ts.index.to_frame(index=False, name=names)
    ts = ts.reset_index(drop=True)
    decoded = df.code.apply(decoder, pad_size=pad_size, variables=variables).apply(pd.Series)
    decoded.columns = variables
    df = pd.concat([df[index], decoded, ts], axis=1)
    index += variables
    df = df.set_index(index)
    return df


def find_pad_size(df, only_max=False):
    """
    finds the (maximum?) number of digits needed to represent a set of columns made up of integers using a string encoding
    """
    if type(df) is not pd.DataFrame:
        if df.dtype in ['int64', 'float32']:
            sizes = len(str(df.max()))
        else:
            sizes = df.astype(str).map(len).max()
    else:
        sizes = df.apply(lambda x: x.astype(str).map(len).max()).to_list()
        if only_max:
            sizes = [max(sizes)] * len(df.columns)

    return sizes


def extract_hts_residuals(fit, columns, years):
    """
    takes the wrapped output of hts.forecast in R, extracts the residuals, and turns it into a pandas dataframe that looks like the input data
    """
    new_resid = fit.rx('resid')[0]
    new_resid = pd.DataFrame(new_resid, columns=columns)
    new_resid = new_resid[~new_resid.index.duplicated(keep='first')]
    new_resid['year_id'] = years
    # TODO fix time index
    new_resid = new_resid.set_index('year_id')

    return new_resid


def extract_hts_forecast(fit, columns, years):
    """
    takes the wrapped output of hts.forecast in R and turns it into a pandas dataframe that looks like the input data
    """
    forecast = fit.rx('bts')[0]
    backcast = fit.rx('fitted')[0]
    preds = np.append(backcast, forecast, axis=0)
    preds = pd.DataFrame(preds, columns=columns)
    preds = preds[~preds.index.duplicated(keep='first')]
    # TODO fix time index
    preds['year_id'] = years
    preds = preds.set_index('year_id')

    return preds


def da_to_encoded_df(da, name, variables, to_drop, index, **kwargs):
    """
    takes a data array and returns a flat encoded dataframe of the sort taken by the hts package
    """
    df = da.to_dataframe(name)
    df = df.reset_index()
    pad_size = find_pad_size(df[variables])

    return to_flat_hierarchical_df(df, name=name, to_drop=to_drop, index=index, variables=variables, pad_size=pad_size, **kwargs)


def get_cv_blocks(start=2000, end=2017, size=5):
    # TODO fix time index
    return [[y for y in range(year, min([end, year + size]))] for year in range(start, end)]


def filter_by_code(d, variables, code, variable_types):
    filter_values = {variables[i]: code[i] for i in range(0, len(code))}
    filter_values = pd.DataFrame([filter_values])
    filter_values = filter_values.astype(variable_types)
    filter_values = list(filter_values.itertuples(index=False, name=None))[0]
    ret = d.loc[filter_values]
    ret = np.asarray(ret).T
    return ret


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
