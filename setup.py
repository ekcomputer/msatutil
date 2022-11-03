import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)

setup(
    name='msatutil',
    desciption='Utility codes to read and plot from MethaneSAT/AIR files',
    author='Sebastien Roche',
    author_email='sroche@g.harvard.edu',
    version='1.0.0',  # make sure stays in sync with the version in msatutil/__init__.py
    url='',
    install_requires=[
        'dask',
        'netcdf4',
        'matplotlib',
        'pandas',
        'scipy',
    ],
    packages=find_packages(),
    include_package_data=True,
)
