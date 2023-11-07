import os
from setuptools import setup, find_packages

_mydir = os.path.dirname(__file__)

# extra dependencies for mair_geoviews.ipynb
extras = {
    "notebooks": [
        "notebook",
        "panel",
        "holoviews",
        "cartopy",
        "geoviews",
        "datashader",
    ],
}

setup(
    name="msatutil",
    description="Utility codes to read and plot from MethaneSAT/AIR files",
    author="Sebastien Roche",
    author_email="sroche@g.harvard.edu",
    version="3.14.2",  # make sure stays in sync with the version in msatutil/__init__.py
    url="https://github.com/rocheseb/msatutil",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "dask",
        "netcdf4",
        "matplotlib",
        "pandas",
        "scipy",
        "tqdm",
        "google-cloud-storage",
        ],
    extras_require=extras,
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.9,<3.12",
    entry_points={
        "console_scripts": [
            "mairl3html=msatutil.mair_geoviews:main",
        ],
    },
)
