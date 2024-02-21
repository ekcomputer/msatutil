import os
import platform
from msat_dset import msat_dset
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon, shape
from rasterio.transform import from_origin

## I/O
# L3_mosaics_catalogue_pth = '/Volumes/metis/MAIR/Index/L3_mosaics.xlsx'
L3_mosaics_catalogue_pth = 'gs://msat-dev-science-data/L3_mosaics.csv'
os_platform = platform.platform()

## Load
# gdf = pd.read_excel(L3_mosaics_catalogue_pth)
if os_platform == 'macOS-14.2.1-arm64-arm-64bit':
    df = pd.read_csv(L3_mosaics_catalogue_pth, storage_options={
        'token': '~/.config/gcloud/application_default_credentials_MAIR.json'})
    df = df[:3]  # Testing
    working_dir = '/Volumes/metis/MAIR/Spatial_catalogue'
else:  # on GCS
    df = pd.read_csv(L3_mosaics_catalogue_pth,
                     storage_options={'token': 'cloud'})
    working_dir = '~/msat_spatial_idx'
catalogue_shp_out_pth = os.path.join(working_dir, 'L3_mosaics_20240208.shp')

## TODO: problem is google-auth installation is not working. (Can't import gcsfs). I had google-auth 1.35 installed, then installed 2.28 with pip, but it had a dependency conflict, so I uninstalled. Tried to install with mamba, but it tells me it is already installed, even though import google-auth fails...

## Could also try loading Sebastian's way and using pandas to load 'data'
'''

e.g. with nc_target.open("rb") as gcloud_file:
                data = gcloud_file.read()

    +google-api-core-1.31.5 (conda-forge/noarch)
    +google-api-python-client-1.12.8 (conda-forge/noarch)
    +google-auth-1.35.0 (conda-forge/noarch)
    +google-auth-httplib2-0.1.0 (conda-forge/noarch)
    +google-cloud-core-2.3.1 (conda-forge/noarch)
    +google-cloud-storage-2.10.0 (conda-forge/noarch)
    +google-crc32c-1.1.2 (conda-forge/osx-arm64)
    +google-resumable-media-2.5.0 (conda-forge/noarch)
    +googleapis-common-protos-1.60.0 (conda-forge/noarch)

'''
## Functions

def validDataArea2Gdf(ds, simplify=None):
    '''
    This function converts valid data areas from a msatutil.msat_dset.msat_dset (Netcdf4 Dataset) into a MultiPolygon Shapefly geometry.

    Parameters:
    - ds: Netcdf4 Dataset object containing the data
    - simplify: Not used in the function (default: None)

    Returns:
    - multipolygon: A MultiPolygon object representing the valid data areas

    Notes:
    - The function assumes a specific geospatial resolution of '1/3 arcseconds' and returns an assertion error if this is not true.
    - The function assumes geographic coordinates in WGS84 (EPSG:4326).
    - The function assumes that the dataset has a 'xch4' variable that contains the valid data areas.
    - It creates a MultiPolygon object from valid data areas based on the 'xch4' variable in the dataset.
    - The function uses a transformation defined by the geospatial information of the dataset.

    Examples:
    validDataArea2Gdf(ds, simplify=None)
    '''
    # Define the transform and metadata for the temporary raster
    if ds.geospatial_lat_resolution == ds.geospatial_lon_resolution == '1/3 arcseconds':
        res = 1 / 60 / 60 / 3
    elif ds.geospatial_lat_resolution == ds.geospatial_lon_resolution == '1 arcsecond':
        res = 1 / 60 / 60
    else:
        raise ValueError(
            "Geospatial resolutions of latitude and longitude do not match or value needs to be specified")

    transform = from_origin(float(ds.geospatial_lon_min), float(ds.geospatial_lat_max), res,
                            res)  # Adjust these values

    data_variable = ds.variables['xch4'][:]
    valid_data_mask = ~np.isnan(data_variable)

    ## Convert to geometry
    shapes_gen = shapes((~valid_data_mask.mask).astype(
        'uint8'), transform=transform)

    polygons = []
    for poly_shape, value in shapes_gen:
        if value == 1:  # Valid data value
            polygons.append(shape(poly_shape))

    multipolygon = MultiPolygon(polygons)

    if simplify is not None:
        # Simplify the geometry
        multipolygon = multipolygon.simplify(simplify, preserve_topology=False)
    return multipolygon


if __name__ == '__main__':
    for index, row in df.iterrows():
        gs_pth = row['uri']
        print(gs_pth.split('/mosaic/')[-1])
        ds = msat_dset(gs_pth)
        geom = validDataArea2Gdf(ds, simplify=0.001)
        df.at[index, 'geometry'] = geom

    gdf = gpd.GeoDataFrame(df, geometry='geometry',
                           crs='EPSG:4326')

    ## ESRI shapefile can't handle datetimeformat
    for col in ['flight_date', 'production_timestamp', 'time_start', 'time_end']:
        gdf[col] = gdf[col].astype(str)

    ## Save as shapefile and geojson to disk
    gdf.to_file(catalogue_shp_out_pth)
    gdf.to_file(catalogue_shp_out_pth.replace('.shp', '.geojson'))
