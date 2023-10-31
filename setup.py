import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)

setup(
    name="msatutil",
    description="Utility codes to read and plot from MethaneSAT/AIR files",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    version="3.5.0",  # make sure stays in sync with the version in msatutil/__init__.py
    url="https://github.com/rocheseb/msatutil",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["dask", "netcdf4", "matplotlib", "pandas", "scipy", "tqdm"],
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.9",
)
