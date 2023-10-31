# MSATutil

Utility programs for reading **MethaneSAT/AIR** files and making quick diagnostics plots

# Install

`git clone https://github.com/rocheseb/msatutil`

`pip install -e msatutil`

To be able to run the [mair_geoviews](notebooks/mair_geoviews.ipynb) notebook use this command instead:

`pip install -e msatutil[notebooks]`

# msatutil

## msatutil.msat_dset

Just extends netCDF4.Dataset to allow opening files in the google cloud storage starting with **gs://**

### msatutil.msat_nc

The **msat_nc** class represents a single L1/L2/L2-pp/L3 file with added metadata and convenience methods for browsing the data

Most useful methods are:

* msat_nc.show_all: show all the variables in the file and their dimensions
* msat_nc.search: search for a given keyword amongst the variables in the file
* msat_nc.fetch: fetch the first variable data that corresponds to the given keyword
* msat_nc.show_sv: for L2 files, show the state vector metadata
* msat_nc.fetch_varpath: get the full variable path that corresponds to the given keyword

#### msatutil.msat_interface

The most important object here is the **msat_collection** class which can open a list of L1/L2/L3 files

Its most useful methods are:

* msat_collection.pmesh_prep: returns a given variable from all the files concatenated along-track

* msat_collection.grid_prep: returns rough L3, the given variable on a regular lat/lon grid using mean aggregation for overlaps

* msat_collection.heatmap: use matplotlib's pcolormesh to plot the given variable either in along-/across-track indices or in lat/lon

* msat_collection.hist: makes a histogram of the given variable

It has most of the **msat_nc** convenience methods

There is a **get_msat** function to generate a **msat_collection** from all the files in a directory


### msatutil.compare_heatmaps

Can be used to compare granules from two different **msat_collection** objects

# Notebooks

There are notebooks showing example usage of the msatutil programs:

[msat_interface](notebooks/msat_interface_example.ipynb)

[compare_heatmaos](notebooks/compare_heatmaps_example.ipynb)

[mair_geoviews](notebooks/mair_geoviews.ipynb)

### Running mair_geoviews.ipynb

[mair_geoviews](notebooks/mair_geoviews.ipynb) is for plotting L3 data or a full flight worth of L1/L2/L2-pp data using Holoviz libraries

#### with a local webserver

e.g. from the parent directory of the cloned msatutil repo:

`panel serve --show msatutil/notebooks/mair_geoviews.ipynb`

#### with jupyter

`jupyter notebook msatutil/notebooks/mair_geoviews.ipynb`
