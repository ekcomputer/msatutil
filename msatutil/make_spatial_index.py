import os
import platform
from msat_dset import msat_dset
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape
from rasterio.transform import from_origin

## Functions

def validDataArea2Gdf(ds, simplify=None):
    '''
    This function converts valid data areas from a msatutil.msat_dset.msat_dset (Netcdf4 Dataset) into a MultiPolygon Shapefly geometry. It takes about an hour to run for 1,000 mosaics.

    See https://github.com/methanesat-org/sandbox-viz-app/blob/main/src/make_geotiff_L3.py for a similar script that converts netcdf to geotiff.

    Parameters:
    - ds: Netcdf4 Dataset object containing the data
    - simplify: Not used in the function (default: None)

    Returns:
    - multipolygon: A MultiPolygon object representing the valid data areas

    Notes:
    - The function assumes a specific geospatial resolution of 1/3 or 1arcseconds and returns an assertion error if this is not true.
    - The function assumes geographic coordinates in WGS84 (EPSG:4326).
    - The function assumes that the dataset has a 'xch4' variable that contains the valid data areas.
    - It creates a MultiPolygon object from valid data areas based on the 'xch4' variable in the dataset.
    - The function uses a transformation defined by the geospatial information of the dataset.
    - There were some errors using a simplify_tol of 0.001 degrees. Using 0.01 instead is slightly too coarse, but saves disk space and makes rendering quick.

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
    transform = from_origin(float(ds.geospatial_lon_min), float(ds.geospatial_lat_min), res,
                            -res)  # Adjust these values
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
    ## I/O
    L3_mosaics_catalogue_pth = 'gs://msat-dev-science-data/L3_mosaics.csv'
    os_platform = platform.platform()
    load_from_chkpt = True
    simplify_tol = None # in map units (deg)

    ## Load
    if simplify_tol is not None:
        tol_str = f'_tol{simplify_tol}'
    else:
        tol_str = ''
    if os_platform == 'macOS-14.2.1-arm64-arm-64bit':
        working_dir = '/Volumes/metis/MAIR/Spatial_catalogue' 
        catalogue_shp_out_pth = os.path.join(working_dir, f'L3_mosaics_20240208{tol_str}.shp')
        if load_from_chkpt and os.path.exists(os.path.expanduser(catalogue_shp_out_pth)):
            df = gpd.read_file(catalogue_shp_out_pth, driver='ESRI Shape')
        else:
            df = pd.read_csv(L3_mosaics_catalogue_pth)            
        df = df[:3]  # Testing
    else:  # on GCS 'Linux-6.2.0-1013-gcp-x86_64-with-glibc2.35'
        working_dir = '~/msat_spatial_idx'
        catalogue_shp_out_pth = os.path.join(working_dir, f'L3_mosaics_20240208{tol_str}.shp')
        if load_from_chkpt and os.path.exists(os.path.expanduser(catalogue_shp_out_pth)):
            df = gpd.read_file(catalogue_shp_out_pth, driver='ESRI Shape')
        else:
            df = pd.read_csv(L3_mosaics_catalogue_pth,
                            storage_options={'token': 'cloud'})

    ## Loop
    for index, row in df.iterrows():
        gs_pth = row['uri']
        if 'geometry' in df.columns:  # loaded from checkpoint
            if  pd.notnull(df.at[index, 'geometry']): 
                print(f"Geometry exists for {gs_pth.split('/mosaic/')[-1]}")
                continue  # Skip if geometry is already present

        print(f"[{index}] {gs_pth.split('/mosaic/')[-1]}")
        ds = msat_dset(gs_pth)
        geom = validDataArea2Gdf(ds, simplify=simplify_tol)
        df.at[index, 'geometry'] = geom

        ## Save intermittently
        if (index % 50 == 0) or index == len(df):
            if index % 10 == 0:
                print('\t> Saving checkpoint.')
            if index == len(df) - 1:
                print('\t> Saving final.')
            gdf = gpd.GeoDataFrame(df, geometry='geometry',
                                crs='EPSG:4326')

            ## ESRI shapefile can't handle datetimeformat
            try:
                for col in ['flight_date', 'production_timestamp', 'time_start', 'time_end']:
                    gdf[col] = gdf[col].astype(str)
            except:
                for col in ['flight_dat', 'producti_2', 'time_start', 'time_end']:
                    gdf[col] = gdf[col].astype(str)
            
            ## Save as shapefile and geojson to disk
            gdf.to_file(catalogue_shp_out_pth, driver='ESRI Shapefile')
            gdf.to_file(catalogue_shp_out_pth.replace('.shp', '.geojson'))
    print('Finished creating mask shapefile.')