Python wrapper for Hyndman's forecast package in R

designed to be convenient in cases where there are many stacked timeseries and you want to fit the same model type to all series (e.g. hierarchical, grouped time-series). You can simply pass in an xarray or dataframe along with the relevant grouping variables and get back point forecasts, uncertainty intervals, or draws from the forecast distribution.

primary model types that are supported are exponential smoothing and ARIMA with automatic tuning (or manual setting) of the relevant hyperparameters.

there is also some support for pre-model fitting scaling and transformation.
