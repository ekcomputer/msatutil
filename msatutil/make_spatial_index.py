import os
from msatutil.msat_dset import msat_dset
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon, shape
from rasterio.transform import from_origin

## I/O
L3_mosaics_catalogue_pth = '/Volumes/metis/MAIR/Index/L3_mosaics.xlsx'
working_dir = '/Volumes/metis/MAIR/Spatial_catalogue'
catalogue_shp_out_pth = os.path.join(working_dir, 'L3_mosaics_20240208.shp')

## I/O
L3_mosaics_catalogue_pth = '/Volumes/metis/MAIR/Index/L3_mosaics.xlsx'
working_dir = '/Volumes/metis/MAIR/Spatial_catalogue'
catalogue_shp_out_pth = os.path.join(working_dir, 'L3_mosaics_20240208.shp')

## Load
gdf = pd.read_excel(L3_mosaics_catalogue_pth)

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
    gdf = gdf[:3]  # Testing
    gdf
    for index, row in gdf.iterrows():
        gs_pth = row['uri']
        print(gs_pth.split('/mosaic/')[-1])
        ds = msat_dset(gs_pth)
        geom = validDataArea2Gdf(ds, simplify=0.001)
        gdf.at[index, 'geometry'] = geom

    gdf
    gdf.crs = 'EPSG:4326'
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry',
                           crs='EPSG:4326')  # , crs='EPSG:4326')
    for col in ['flight_date', 'production_timestamp', 'time_start', 'time_end']:
        gdf[col] = gdf[col].astype(str)
    gdf.to_file(catalogue_shp_out_pth)
