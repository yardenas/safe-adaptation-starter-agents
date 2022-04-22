#!/usr/bin/env python

from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Safety Starter Agents is designed to work with Python 3.6 and greater. " \
    + "Please install it before proceeding."

setup(
    name='safe_rl',
    packages=['safe_rl'],
    install_requires=[
        'joblib',
        'matplotlib==3.1.1',
        'mpi4py',
        'numpy~=1.22.3',
        'seaborn==0.8.1',
        'tensorflow',
    ],
)
