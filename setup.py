#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='forecastr',
    version='0.0.1',
    url='https://github.com/zmjones/forecastr',
    description='wrapper for some functions in hyndmans r package',
    author='Zachary Jones',
    author_email='zmjone2992@gmail.com',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'xarray', 'scipy', 'rpy2'],
)
