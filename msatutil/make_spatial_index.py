import warnings
import os
import argparse
from msatutil.msat_dset import msat_dset
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape
from rasterio.transform import from_origin

## Use pyogrio for reading/writing large shapefiles
engine = 'pyogrio'


# Ignore UserWarning raised by google.auth._default
warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials from Google Cloud SDK without a quota project.",
    category=UserWarning,
    module="google.auth._default"
)

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
        try:
            multipolygon = multipolygon.simplify(
                simplify, preserve_topology=False)
        except:
            ## Sometimes the simplify operation can return an error
            pass
    return multipolygon


def save_geojson(L3_mosaics_catalogue_pth: str, working_dir: str, load_from_chkpt: bool = True, simplify_tol: float = None, save_frequency: int = 50, out_path: str = None) -> list[str]:
    '''
    save_geojson Calls validDataArea2Gdf and writes out as an ESRI shapefile (if no polygon simplification), or a geojson otherwise.

    Requires user to be authenticated to GCS in order to load cloud paths.

    Parameters
    ----------
    L3_mosaics_catalogue_pth : str
        Local or cloud path to a csv file with a column 'uri' listing cloud paths to L2 or L3 .nc files. Other attributes are copied over to final output. If using a cloud path, requires gcsfs to be installed.
    working_dir : str
        Output directory to save final spatial file
    load_from_chkpt : bool, optional
        Whether to load from an existing (potentially partly complete) output, based on name, or from `L3_mosaics_catalogue_pth`, by default True
    simplify_tol : float, optional
        If given, uses polygon simplification of `simplify_tol` map units to reduce output file size and rendering times, by default None
    save_frequency : int, optional
        Enables intermediate saving every `save_frequency` files. Set to a high number to disable. B default 50
    out_path : str, optional
        Output path (extensions will be replaced), by default uses basename of `L3_mosaics_catalogue_pth`
    '''
    ## Try-except block allows function to return output path names during testing, even if there is a keyboard interrupt.
    try:
        ## Load
        if simplify_tol is not None:
            tol_str = f'_tol{simplify_tol}'
        else:
            tol_str = ''
        if out_path is None:
            catalogue_shp_out_basename = os.basename(
                L3_mosaics_catalogue_pth).replace('.csv', '')
        else:
            catalogue_shp_out_basename = os.path.splitext(out_path)[0]
        catalogue_shp_out_pth = os.path.join(
            working_dir, f'{catalogue_shp_out_basename}{tol_str}.shp')
        catalogue_geojson_out_pth = catalogue_shp_out_pth.replace(
            '.shp', '.geojson')
        if load_from_chkpt and os.path.exists(os.path.expanduser(catalogue_shp_out_pth)):
            df = gpd.read_file(catalogue_shp_out_pth, engine=engine)
        else:
            # storage_options={'token': 'cloud'}
            df = pd.read_csv(L3_mosaics_catalogue_pth)

        ## Loop
        for index, row in df.iterrows():
            gs_pth = row['uri']
            if 'geometry' in df.columns:  # loaded from checkpoint
                if pd.notnull(df.at[index, 'geometry']):
                    print(
                        f"Geometry exists for {gs_pth.split('/mosaic/')[-1]}")
                    continue  # Skip if geometry is already present

            print(f"[{index}] {gs_pth.split('/mosaic/')[-1]}")
            ds = msat_dset(gs_pth)
            geom = validDataArea2Gdf(ds, simplify=simplify_tol)
            df.at[index, 'geometry'] = geom

            ## Save intermittently
            if (index % save_frequency == 0) or (index == len(df) - 1):
                if (index % save_frequency == 0) and (index != len(df) - 1):
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
                gdf.to_file(catalogue_shp_out_pth, engine=engine)
                if simplify_tol is not None:
                    gdf.to_file(catalogue_geojson_out_pth)
                else:
                    catalogue_geojson_out_pth = ''
        print('Finished creating mask shapefile.')
    except KeyboardInterrupt:
        pass
    finally:
        return catalogue_shp_out_pth, catalogue_geojson_out_pth


def main():
    parser = argparse.ArgumentParser(
        description="Calls validDataArea2Gdf and writes out as an ESRI shapefile (if no polygon simplification), or a geojson otherwise. Requires user to be authenticated to GCS in order to load cloud paths.")

    # Required arguments with shortcuts
    parser.add_argument("-c", "--L3_mosaics_catalogue_pth", type=str, required=True,
                        help="Local or cloud path to a csv file listing cloud paths to L2 or L3 .nc files. Requires gcsfs for cloud paths.")
    parser.add_argument("-w", "--working_dir", type=str, required=True,
                        help="Output directory to save the final spatial file.")

    # Optional arguments with shortcuts
    parser.add_argument("-l", "--load_from_chkpt", type=bool, default=True,
                        help="Load from an existing output or from the catalogue path. Default: True.")
    parser.add_argument("-s", "--simplify_tol", type=float, default=None,
                        help="Polygon simplification tolerance in map units to reduce file size and rendering times. Default: None.")
    parser.add_argument("-f", "--save_frequency", type=int, default=50,
                        help="Intermediate saving frequency every 'n' files. High number disables it. Default: 50.")
    parser.add_argument("-o", "--out_path", type=str, default=None,
                        help="Output path for the file, with extensions replaced. Default: Basename of L3_mosaics_catalogue_pth.")

    args = parser.parse_args()

    # Call the function with the parsed arguments
    save_geojson(
        L3_mosaics_catalogue_pth=args.L3_mosaics_catalogue_pth,
        working_dir=args.working_dir,
        load_from_chkpt=args.load_from_chkpt,
        simplify_tol=args.simplify_tol,
        save_frequency=args.save_frequency,
        out_path=args.out_path,
    )


if __name__ == '__main__':
    main()
