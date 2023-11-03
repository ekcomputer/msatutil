import warnings

warnings.simplefilter("ignore")
import os
import numpy as np
import argparse
import holoviews as hv
from holoviews.operation.datashader import rasterize
import geoviews as gv
from geoviews.tile_sources import EsriImagery

from msatutil.msat_dset import msat_dset

from typing import Optional, Tuple

import subprocess

hv.extension("bokeh")


def show_map(
    x,
    y,
    z,
    width: int = 450,
    height: int = 450,
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    alpha: int = 1,
    title: str = "",
    background_tile=EsriImagery,
):
    """
    Make a geoviews map of z overlayed on background_tile
    This doesn't preserve pixel shapes as the image is re-rasterized after each zoom/pan

    Inputs:
        x: 1D or 2D array of longitudes (shape (N,) or (M,N))
        y: 1D or 2D array of latitudes (shape (M,) or (M,N))
        z: 2D array of the data to plot (shape (M,N))
        width (int): plot width in pixels
        height (int): plot height in pixels
        cmap (str): named colormap
        clim (Optional[Tuple[float, float]]): z-limits for the colorbar, give False to use dynamic colorbar
        alpha (float): between 0 and 1, sets the opacity of the plotted z field
        title (str): plot title
        background_tile: the geoviews tile the plot will be made over and that will be in the linked 2nd panel
                         if None only makes one panel with no background but with the save tool active

    Outputs:
        geoviews figure
    """

    quad = gv.project(gv.QuadMesh((x, y, z)))

    if clim is None:
        # define color limits as mean +/- 3 std
        mean_z = np.nanmean(z)
        std_z = np.nanstd(z, ddof=1)
        clim = (mean_z - 3 * std_z, mean_z + 3 * std_z)

    raster = rasterize(quad, width=width, height=height).opts(
        width=width,
        height=height,
        cmap=cmap,
        colorbar=True,
        title=title,
        alpha=alpha,
    )

    if clim is not False:
        raster = raster.opts(clim=clim)

    if background_tile is not None:
        # Make a dummy quadmesh that will have alpha=0 in the second panel so we can see the EsriImagery under the data
        # I do this so it will add a colorbar on the second plot so we don't need to think about resizing it
        # just use a small subset of data so it doesn't trigger much computations
        if x.ndim == 1:
            xdum = x[:10]
            ydum = y[:10]
        else:
            xdum = x[:10, :10]
            ydum = y[:10, :10]

        dummy = gv.project(gv.QuadMesh((xdum, ydum, z[:10, :10]))).opts(
            width=width,
            height=height,
            cmap=cmap,
            colorbar=True,
            alpha=0,
            title="Esri Imagery",
        )
        if clim is not False:
            dummy.opts(clim=clim)

    if background_tile is None:
        plot = raster
    else:
        plot = background_tile * (raster + dummy)

    return plot


def do_single_map(
    mosaic_file: str,
    var: str,
    out_path: Optional[str] = None,
    title="",
    cmap: str = "viridis",
    width: int = 850,
    height: int = 750,
    alpha: float = 1,
) -> None:
    """
    Save a html plot of var from mosaic_file

    mosaic_file (str): full path to the input L3 mosaic file, can be a gs:// path
    var (str): variable name in the L3 mosaic file
    out_path (Optional[str]): full path to the output .html file or to a directory where it will be saved.
                              If None, save output html file in the current working directory
    title (str): title of the plot (include field name and units here)
    cmap (str): colormap name
    width (int): plot width in pixels
    height (int): plot height in pixels
    """
    default_html_filename = os.path.basename(mosaic_file).replace(".nc", ".html")
    if out_path is None:
        out_file = os.path.join(os.getcwd(), default_html_filename)
    elif os.path.splitext(out_path) == "":
        os.makedirs(out_path)
        out_file = os.path.join(out_path, default_html_filename)
    else:
        os.makedirs(os.path.dirname(out_path))
        out_file = out_path

    with msat_dset(mosaic_file) as nc:
        lon = nc["lon"][:]
        lat = nc["lat"][:]
        v = nc[var][:]

    plot = show_map(lon, lat, v, title=title, cmap=cmap, width=width, height=height, alpha=alpha)

    hv.save(plot, out_file, backend="bokeh")

    print(out_file)


def L3_mosaics_to_html(
    l3_dir: str,
    out_dir: str,
    var: str = "xch4",
    overwrite: bool = False,
    html_index: bool = False,
    title_prefix: str = "",
    cmap: str = "viridis",
    width: int = 850,
    height: int = 750,
    alpha: float = 1,
) -> None:
    """
    l3_dir: full path to the L3 directory, assumes the following directory structure

    l3_dir
    -target_dir
    --resolution_dir
    ---mosaic_file

    out_dir (str): full path to the directory where the plots will be saved
    overwrite (bool): if True overwrite existing plots if they have the same name
    html_index (bool): if True, will generated index.html files in the output directory tree

    plot parameters:

    var (str): name of the variable to plot from the L3 files
    title_prefix (str): plot titles will be "title_prefix; target; resolution; XCH4 (pbb)"
    cmap (str): name of the colormap
    width (int): plot width in pixels
    height (int): plot height in pixels
    """
    target_dir_list = os.listdir(l3_dir)

    for target in target_dir_list:
        print(target)
        target_dir = os.path.join(l3_dir, target)
        resolution_dir_list = os.listdir(target_dir)
        for resolution in resolution_dir_list:
            if resolution == "10m":
                continue
            print(f"\t{resolution}")
            resolution_dir = os.path.join(target_dir, resolution)
            mosaic_file_list = os.listdir(resolution_dir)
            for mosaic_file in mosaic_file_list:
                print(f"\t\t{mosaic_file}")

                plot_out_dir = os.path.join(out_dir, target, resolution)
                if not os.path.exists(plot_out_dir):
                    os.makedirs(plot_out_dir)
                plot_out_file = os.path.join(plot_out_dir, mosaic_file.replace(".nc", ".html"))

                if os.path.exists(plot_out_file) and not overwrite:
                    continue

                mosaic_file_path = os.path.join(resolution_dir, mosaic_file)

                title = (
                    f"{title_prefix}; {' '.join(target.split('_')[1:])}; {resolution}; XCH4 (ppb)"
                )

                do_single_map(
                    mosaic_file_path,
                    var,
                    out_file=plot_out_file,
                    title=title,
                    cmap=cmap,
                    width=width,
                    height=height,
                    alpha=alpha,
                )

    if html_index:
        generate_html_index(out_dir)


def generate_html_index(out_dir: str) -> None:
    """
    Uses tree to recursively generated index.html file under out_dir
    These link to all the *html files in out_dir

    out_dir (str): the directory to be recursively indexed
    """

    subprocess.run(
        ["sh", os.path.join(os.path.dirname(__file__), "generate_html_index.sh"), out_dir]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "l3_path",
        help="full path to the directory with L3 mosaic data or to a L3 mosaic file",
    )
    parser.add_argument(
        "out_path",
        help="full path to the output directory",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="if given, overwrite existing plots",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Will be added to the plot titles",
    )
    parser.add_argument(
        "-i",
        "--index",
        action="store_true",
        help="if given, will generate index.html files in the output directory tree",
    )
    parser.add_argument(
        "-c",
        "--cmap",
        default="viridis",
        help="colormap name",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=850,
        help="width of the plots in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=750,
        help=" heigh of the plots in pixels",
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="alpha value (transparency) for the plot, between 0 (transparent) and 1 (opaque)",
    )
    parser.add_argument(
        "-v",
        "--variable",
        default="xch4",
        help="name of the variable to plot from the L3 files",
    )
    args = parser.parse_args()

    if os.path.splitext(args.l3_path)[1] != "":
        # If l3_path point directly to a L3 mosaic file
        do_single_map(
            args.l3_path,
            args.variable,
            out_path=args.out_path,
            title=args.title,
            cmap=args.cmap,
            width=args.width,
            height=args.height,
            alpha=args.alpha,
        )
    else:
        # If l3_path point to a directory
        L3_mosaics_to_html(
            args.l3_path,
            args.out_path,
            var=args.variable,
            overwrite=args.overwrite,
            html_index=args.index,
            title_prefix=args.title,
            cmap=args.cmap,
            width=args.width,
            height=args.height,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
