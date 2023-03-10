from __future__ import annotations
import os
import sys
import glob
import numpy as np
import netCDF4 as ncdf
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import argparse
from msatutil.msat_nc import msat_nc
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union, Annotated, List
import dask
import dask.array as da
import time
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import pickle


def meters_to_lat_lon(x: float, lat: float) -> float:
    """
    Convert a distance in meters to latitudinal and longitudinal angles at a given latitude
    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    Uses WGS84 https://en.wikipedia.org/wiki/World_Geodetic_System

    Inputs:
        x: distance (meters)
        lat: latitude (degrees)
    Outputs:
        (lat_deg,lon_deg): latitudinal and longitudinal angles corresponding to x (degrees)

    """
    lat = np.deg2rad(lat)
    meters_per_lat_degree = (
        111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    )

    meters_per_lon_degree = (
        111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    )

    return x / meters_per_lat_degree, x / meters_per_lon_degree


def filter_large_triangles(points: np.ndarray, tri: Optional[Delaunay] = None, coeff: float = 2.0):
    """
    Filter out triangles that have an edge > coeff * median(edge)
    Inputs:
        tri: scipy.spatial.Delaunay object
        coeff: triangles with an edge > coeff * median(edge) will be filtered out
    Outputs:
        valid_slice: boolean array that select for the triangles that
    """
    if tri is None:
        tri = Delaunay(points)

    edge_lengths = np.zeros(tri.vertices.shape)
    seen = {}
    # loop over triangles
    for i, vertex in enumerate(tri.vertices):
        # loop over edges
        for j in range(3):
            id0 = vertex[j]
            id1 = vertex[(j + 1) % 3]

            # avoid calculating twice for non-border edges
            if (id0, id1) in seen:
                edge_lengths[i, j] = seen[(id0, id1)]
            else:
                edge_lengths[i, j] = np.linalg.norm(points[id1] - points[id0])

                seen[(id0, id1)] = edge_lengths[i, j]

    median_edge = np.median(edge_lengths.flatten())

    valid_slice = np.all(edge_lengths < coeff * median_edge, axis=1)

    return valid_slice


def chunked(lst: List, n: int):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"{func.__name__} done in {time.time()-start} s")

    return wrapper


def get_msat(indir, srchstr="Methane*.nc"):
    """
    Function to get the L1B or L2 files under indir into a msat_collection object
    """
    flist = glob.glob(os.path.join(indir, srchstr))
    return msat_collection(flist, use_dask=True)


class msat_file(msat_nc):
    """
    Class to interface with a single MethaneSAT/AIR L1B or L2 file
    It has methods to make simple plots.
    """

    def __init__(self, msat_file: str, use_dask: bool = False) -> None:
        super().__init__(msat_file, use_dask=use_dask)

        try:  # this is a try so we can use the same class to read the L1 files
            sp_slice = self.get_sv_slice("SurfacePressure")
            if not self.use_dask:
                self.dp = (
                    self.nc_dset["SpecFitDiagnostics"]["APosterioriState"][sp_slice]
                    - self.nc_dset["SpecFitDiagnostics"]["APrioriState"][sp_slice]
                )
            else:
                self.dp = da.from_array(
                    self.nc_dset["SpecFitDiagnostics"]["APosterioriState"][sp_slice]
                ) - da.from_array(self.nc_dset["SpecFitDiagnostics"]["APrioriState"][sp_slice])
        except:
            pass

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.nc_dset.close()

    def get_var(self, var: str, grp: Optional[str] = None, chunks: Union[str, Tuple] = "auto"):
        """
        return a variable array from the netcdf file
        var: complete variable name
        grp: complete group name
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        if var == "dp":
            return self.dp
        else:
            return super().get_var(var, grp=grp, chunks=chunks)

    def fetch_varpath(self, key, grp: Optional[str] = None) -> str:
        return super().fetch_varpath(key, grp=grp)

    def hist(
        self, ax: plt.Axes, grp: str, var: str, label: str, color: Optional[str] = None
    ) -> None:
        """
        Plot a histogram of the given variable
        ax: matplotlib axes object
        grp: complete group name
        var: complete variable name
        label: horizontal axis label
        color: the color of the bars
        """
        if var == "dp":
            flat_var = self.dp.flatten()
        else:
            flat_var = self.nc_dset[grp][var][:].flatten()
        var_mean = flat_var.mean()
        var_std = flat_var.std(ddof=1)
        if color is None:
            line = ax.axvline(x=var_mean, color=color, linestyle="--")
            ax.hist(
                flat_var,
                edgecolor=line.get_color(),
                facecolor="None",
                label=f"{label}{var_mean:.2e}$\pm${var_std:.2e}",
            )
        else:
            ax.hist(
                flat_var,
                edgecolor=color,
                facecolor="None",
                label=f"{label}{var_mean:.2e}$\pm${var_std:.2e}",
            )
            ax.axvline(x=var_mean, color=color, linestyle="--")

    def spec_plot(self, ax: plt.Axes, j: int, i: int, label: str) -> None:
        """
        Make a spectrum+residuals plot for a given pixel
        ax: a list of 2 matplotlib axes
        j: along-track pixel index
        i: cross-track pixel index
        label: label for the legend
        """
        ax[0].axhline(y=0, linestyle="--", color="black")
        line = self.plot_var(ax[0], "Posteriori_RTM_Band1", "ResidualRadiance", j, i, label)
        self.plot_var(
            ax[1],
            "Posteriori_RTM_Band1",
            "ObservedRadiance",
            j,
            i,
            "Obs",
            color=line.get_color(),
        )
        self.plot_var(
            ax[1],
            "Posteriori_RTM_Band1",
            "Radiance_I",
            j,
            i,
            "Calc",
            color=line.get_color(),
            linestyle="--",
        )
        ax[0].get_shared_x_axes().join(ax[0], ax[1])

    def plot_var(
        self,
        ax: plt.Axes,
        grp: str,
        var: str,
        j: int,
        i: int,
        label: str,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> matplotlib.lines.Line2D:
        """
        Plot a given variable for a given pixel
        ax: matplotlib axes object
        grp: complete group name
        var: complete variable name
        j: along-track pixel index
        i: cross-track pixel index
        label: label for the legend
        color: line color
        linestyle: matplotlib linestyle
        """
        nc_grp = self.nc_dset[grp]
        if nc_grp[var].shape[0] == 1024:
            if nc_grp[var].dimensions[0].startswith("wmx"):
                obs_rad = nc_grp["ObservedRadiance"][:, j, i]
            elif nc_grp[var].dimensions[0].startswith("w1"):  # Native L1 files
                obs_rad = nc_grp["Radiance"][:, j, i]
            obs_rad = np.ma.masked_where(obs_rad == 0, obs_rad)
            if not obs_rad.mask:
                obs_rad.mask = np.zeros(obs_rad.size, dtype=bool)
            if var == "ResidualRadiance":
                rms = 100 * self.get_pixel_rms(j, i)
                sp_slice = self.get_sv_slice("SurfacePressure")
                dp = self.get_pixel_dp(j, i)
                label = f"{label}; rms={rms:.4f}; dP={dp:.3f}"
                ax.set_ylabel("Residuals (%)")
                line = ax.plot(
                    nc_grp["Wavelength"][~obs_rad.mask, j, i],
                    100 * nc_grp[var][~obs_rad.mask, j, i] / obs_rad[~obs_rad.mask],
                    label=label,
                )
            else:
                print(obs_rad.mask)
                print(nc_grp[var][~obs_rad.mask, j, i])
                print(nc_grp["Wavelength"][~obs_rad.mask, j, i])
                line = ax.plot(
                    nc_grp["Wavelength"][~obs_rad.mask, j, i],
                    nc_grp[var][~obs_rad.mask, j, i],
                    label=f'{label.split("_")[3]} {j} {i}',
                )
            ax.set_xlabel("Wavelength (nm)")
        elif nc_grp[var].dimensions[0] == "one":
            line = ax.axhline(y=nc_grp[var][:, j, i], label=label)
            ax.set_ylabel()
        elif nc_grp[var].dimensions[0].startswith("lmx"):
            line = ax.plot(
                self.nc_dset[grp][var][:, j, i],
                self.nc_dset[grp]["PressureMid"][:, j, i],
                label=label,
            )
            ax.set_ylabel("Pressure (hPa)")

        if color:
            line[0].set_color(color)
        if linestyle:
            line[0].set_linestyle(linestyle)

        return line[0]

    def get_pixel_dp(self, j: int, i: int) -> float:
        """
        Calculate posterior minus prior surface pressure
        j: along-track pixel index
        i: cross-track pixel index
        """
        sp_slice = self.get_sv_slice("SurfacePressure")
        dp = (
            self.nc_dset["SpecFitDiagnostics"]["APosterioriState"][sp_slice, j, i]
            - self.nc_dset["SpecFitDiagnostics"]["APrioriState"][sp_slice, j, i]
        )

        return dp[0]

    def get_pixel_rms(self, j: int, i: int) -> float:
        """
        j: along-track pixel index
        i: cross-track pixel index
        """
        return self.nc_dset["SpecFitDiagnostics"]["FitResidualRMS"][:, j, i][0]

    def get_sv_slice(self, var: str) -> np.ndarray:
        """
        Get the state vector index for the given variable
        var: complete state vector variable name
        """
        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__

        for key, val in sv_dict.items():
            if key.startswith("SubStateName") and val.strip() == var:
                num = int(key.split("_")[-1]) - 1
                slice = np.arange(sv_dict["SubState_Idx0"][num] - 1, sv_dict["SubState_Idxf"][num])
                break

        return slice

    def show_sv(self) -> None:
        """
        Display the state vector variable names
        """
        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__
        for key, val in sv_dict.items():
            if type(val) == str:
                val = val.strip()
            print(f"{key.strip()}: {val}")


class msat_collection:
    """
    Class to interface with a list of MethaneSAT/AIR L1B or L2 files.
    Its methods help create quick plots for multiple granules.
    The main methods are pmesh_prep, grid_prep, and heatmap
    pmesh_prep: helps read in a given variable from all the files in the collection by concatenating in the along-track dimension
    grid_prep: is similar to pmesh_prep but puts the data onto a regular lat-lon grid
    heatmap: show a pcolormesh plot of the given variable
    """

    def __init__(
        self,
        file_list: str,
        date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
        use_dask: bool = False,
    ) -> None:
        self.set_use_dask(use_dask)
        file_list = np.array(file_list)
        try:
            dates = np.array(
                [
                    datetime.strptime(os.path.basename(file_path).split("_")[3], "%Y%m%dT%H%M%S")
                    for file_path in file_list
                ]
            )
            sort_ids = np.argsort(dates)
            file_list = file_list[sort_ids]
            dates = dates[sort_ids]

            if date_range:
                file_list = file_list[(dates > date_range[0]) & (dates < date_range[1])]
                dates = dates[(dates > date_range[0]) & (dates < date_range[1])]
            self.dates = dates
        except (ValueError, IndexError):
            print("/!\\ The file names do not have the typical methaneair format")
            self.dates = None

        self.file_paths = file_list
        self.file_names = [os.path.basename(msat_file) for msat_file in file_list]
        self.ids = OrderedDict(
            [(i, msat_file_path) for i, msat_file_path in enumerate(self.file_paths)]
        )
        self.ids_rev = {val: key for key, val in self.ids.items()}
        self.msat_files = OrderedDict(
            [(file_path, msat_file(file_path, use_dask=use_dask)) for file_path in file_list]
        )
        self.dsets = {key: val.nc_dset for key, val in self.msat_files.items()}

        self.is_l2 = "Level1" in list(self.dsets.values())[0].groups
        self.valid_xtrack = self.get_valid_xtrack()

        # self.init_plot(1)

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        for f in self.msat_files.values():
            f.close()

    def get_valid_xtrack(self):
        """
        Get the valid cross track indices
        """
        check_dset = list(self.dsets.values())[0]  # just use any of the datasets
        if self.is_l2:
            select = slice(None) if "one" not in check_dset[f"Level1/Longitude"].dimensions else 0
            valid_xtrack = np.where(
                ~np.isnan(np.nanmedian(check_dset[f"Level1/Longitude"][select], axis=0))
            )[0]
        else:
            if "spectral_channel" in check_dset["Band1/Radiance"].dimensions:
                spec_axis = check_dset["Band1/Radiance"].dimensions.index("spectral_channel")
            else:
                spec_axis = check_dset["Band1/Radiance"].dimensions.index("w1")
            valid_xtrack = np.where(
                np.nanmedian(np.nansum(check_dset[f"Band1/Radiance"][:], axis=spec_axis), axis=0)
                > 0
            )[0]
        valid_xtrack_slice = slice(valid_xtrack[0], valid_xtrack[-1] + 1)

        return valid_xtrack_slice

    def subset(
        self,
        ids: Optional[list] = None,
        date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
        use_dask: bool = True,
    ) -> msat_collection:
        """
        Return a subset msat_collection object corresponding to the ids given (must be present in the keys of self.ids)
        """

        if date_range is not None:
            ids = np.where((self.dates >= date_range[0]) & (self.dates < date_range[1]))[0]

        return msat_collection([self.file_paths[i] for i in ids], use_dask=use_dask)

    def init_plot(self, nplots: int, ratio=[]) -> None:
        """
        Generate an empty figure to be filled by the other methods of this class
        """
        if not ratio:
            ratio = [1 for i in range(nplots)]
        self.fig, self.ax = plt.subplots(nplots, gridspec_kw={"height_ratios": ratio})
        self.lines = {}

        self.fig.set_size_inches(10, 8)

    def plot_var(
        self,
        grp: str,
        var: str,
        j: int,
        i: int,
        ids: Optional[List[int]] = None,
        ax=None,
    ) -> None:
        """
        Plot a given variable at a given pixel
        grp: complete group name
        var: complete variable name
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        ax: matplotlib axes object, if not specified it will use self.ax
        """
        if ax is None:
            ax = self.ax
        if ids is None:
            ids = self.ids.keys()
        ax.grid(True)
        if hasattr(self.msat_files[self.ids[0]].nc_dset[grp][var], "units"):
            ax.set_ylabel(f"{grp} {var} {self.msat_files[self.ids[0]].nc_dset[grp][var].units}")
        else:
            ax.set_ylabel(f"{grp} {var}")

        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            self.msat_files[msat_file].plot_var(ax, grp, var, j, i, label=msat_file)
            self.lines[msat_file] = ax.lines[-1]

        box = ax.get_position()
        if box.width == 0.775:
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc="center left", bbox_to_anchor=[1.04, 0.5], borderaxespad=0)

    def rm_line(self, msat_file, ax=None) -> None:
        """
        Remove a line corresponding to the given msat_file
        msat_file: an msat_file object (from the keys of self.msat_files)
        ax: matplotlib axes object, if not specified it will use self.ax
        """
        if ax is None:
            ax = self.ax
        ax.lines.remove(self.lines[msat_file])

    def pixel_diag(self, j: int, i: int, ids: Optional[List[int]] = None) -> None:
        """
        For a given pixel and file, plot the fit residuals, DP, and the rms of residuals
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        """
        if ids is None:
            ids = self.ids.keys()
        self.init_plot(3)
        self.plot_var("Posteriori_RTM_Band1", "ResidualRadiance", j, i, ids=ids, ax=self.ax[0])
        self.ax[0].set_title("% Residuals")

        # dP and rms plots
        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            dp = self.msat_files[msat_file].get_pixel_dp(j, i)
            line = self.ax[1].axhline(y=dp, label=f"{msat_file}; dP={dp:.3f}")
            line.set_color(self.lines[msat_file].get_color())

            rms = 100 * self.msat_files[msat_file].get_pixel_rms(j, i)
            line = self.ax[2].axhline(y=rms, label=f"{msat_file}; rms={rms:.4f}")
            line.set_color(self.lines[msat_file].get_color())

        self.ax[1].set_ylabel("$\Delta$P (hPa)")
        self.ax[1].set_title("Surface pressure change")
        self.ax[1].grid()

        self.ax[2].set_ylabel("Residual RMS (%)")
        self.ax[2].set_title("RMS of residuals")
        self.ax[2].grid()

        plt.tight_layout()

    def spec_plot(self, j: int, i: int, ids: Optional[List[int]] = None) -> None:
        """
        Make a spectrum+residuals plot for a given pixel in given files
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        """
        if ids is None:
            ids = self.ids.keys()
        self.init_plot(2, ratio=[1, 3])
        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            self.msat_files[msat_file].spec_plot(self.ax, j, i, msat_file)
        for ax in self.ax:
            ax.grid()
            ax.legend()

    def hist(self, grp: str, var: str, ids: Optional[List[int]] = None) -> None:
        """
        Make a histogram for the given variable and given files
        grp: complete group name
        var: complete variable name
        ids: list of ids of the msat files (from the keys of self.ids)
        """
        if ids is None:
            ids = self.ids.keys()
        self.init_plot(1)
        file_list = [self.ids[i] for i in ids]
        for i, msat_file in enumerate(file_list):
            self.msat_files[msat_file].hist(self.ax, grp, var, f"ID={ids[i]}")
        for ax in self.ax:
            ax.grid()
            ax.legend()

    def pmesh_prep(
        self,
        var: str,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis: int = 2,
        chunks: Union[str, Tuple] = "auto",
    ) -> Union[np.ndarray, da.core.Array]:
        """
        get a variable ready to plot with plt.pcolormesh(var)
        var: key contained in the variable to search (uses msat_nc fetch method)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        sv_var: grp will be set to SpecFitDiagnostics and sv_var must be one of APrioriState or APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis: the axis along which the stat is applied (remember the variables are transposed when read)
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        if ids is None:
            ids = self.ids
        else:
            ids = {i: self.ids[i] for i in ids}
        if sv_var is not None and var not in ["APosterioriState", "APrioriState"]:
            raise Exception(
                'var must be one of ["APrioriState","APosterioriState"] when sv_var is given'
            )
        elif sv_var is not None:
            nc_slice = self.get_sv_slice(sv_var)
            grp = "SpecFitDiagnostics"
        else:
            nc_slice = tuple()
        x = []
        for num, i in enumerate(ids.values()):
            if var == "dp":
                x.append(self.msat_files[i].dp.T)
            else:
                if grp is None:
                    x.append(self.msat_files[i].fetch(var, chunks=chunks).T)
                else:
                    x.append(self.msat_files[i].get_var(var, grp, chunks=chunks)[nc_slice].T)
        if len(x[0].shape) == 1:
            axis = 0
        elif "along_track" in self.msat_files[i].nc_dset.dimensions:
            if var == "dp":
                axis = 1
            else:
                varpath = self.fetch_varpath(var.lower(), grp=grp)
                vardim = self.msat_files[i].nc_dset[varpath].dimensions
                ndim = len(vardim)
                axis = vardim.index("along_track")
            if ndim == 2 and axis == 0:
                axis = 1
            elif ndim == 2 and axis == 1:
                axis = 0
            elif ndim == 3 and axis == 0:
                axis = 2
            elif ndim == 3 and axis == 2:
                axis = 0
        else:
            axis = 1

        if self.use_dask:
            x = da.concatenate(x, axis=axis).squeeze()
            x[da.greater(x, 1e29)] = np.nan
        else:
            x = np.concatenate(x, axis=axis).squeeze()
            x[np.greater(x, 1e29)] = np.nan
        if option:
            if self.use_dask:
                x = getattr(da, option)(x, axis=option_axis)
            else:
                x = getattr(np, option)(x, axis=option_axis)
        elif (extra_id is not None) and len(x.shape) == 3:
            x = x[:, :, extra_id]
        elif (extra_id is not None) and len(x.shape) == 4:
            x = x[:, :, extra_id, extra_id]

        if ratio:
            if self.use_dask:
                x = x / da.nanmedian(x)
            else:
                x = x / np.nanmedian(x)
        return x

    def grid_prep(
        self,
        var: str,
        lon_lim: Annotated[Sequence[float], 2],
        lat_lim: Annotated[Sequence[float], 2],
        n: Optional[int] = None,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis: int = 2,
        chunks: Union[str, Tuple] = "auto",
        method: str = "cubic",
        res: float = 20,
    ) -> da.core.Array:
        """
        get a variable ready to plot with plt.pcolormesh(lon_grid,lat_grid,x_grid_avg)
        var: key contained in the variable to search (uses msat_nc fetch method)
        lon_lim: the [min,max] of the longitudes to regrid the data on
        lat_lim: the [min,max] of the latitudes to regrid the data on
        n: the size of chunks to separate the files into (if less than the number of files, there may be white lines in the plot)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        sv_var: grp will be set to SpecFitDiagnostics and sv_var must be one of APrioriState or APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis: the axis along which the stat is applied (remember the variables are transposed when read)
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        method: griddata interpolation method
        res: grid resolution in meters
        """
        if not self.use_dask:
            raise Exception("grid_prep needs self.use_dask==True")

        if ids is None:
            ids = self.ids
        else:
            ids = OrderedDict([(i, self.ids[i]) for i in ids])
        if n is None:
            n = len(ids)

        chunked_ids = list(chunked(list(ids.keys()), n))

        print(
            f"Calling grid_prep on {len(list(ids.keys()))} files, divided in {len(chunked_ids)} chunks of {n} files\n"
        )

        mid_lat = (lat_lim[1] - lat_lim[0]) / 2.0

        lat_res, lon_res = meters_to_lat_lon(res, mid_lat)

        lon_range = da.arange(lon_lim[0], lon_lim[1], lon_res)
        lat_range = da.arange(lat_lim[0], lat_lim[1], lat_res)

        # compute the lat-lon grid now so it doesn't have to be computed for each griddata call
        lon_grid, lat_grid = dask.compute(*da.meshgrid(lon_range, lat_range))

        x_grid_list = []
        for i, ids_slice in enumerate(chunked_ids):
            sys.stdout.write(f"\rgrid_prep now doing chunk {i+1:>3}/{len(chunked_ids)}")
            sys.stdout.flush()

            x = self.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                ids=ids_slice,
                option=option,
                option_axis=option_axis,
                ratio=ratio,
                chunks=chunks,
            )
            lat = self.pmesh_prep("Latitude", ids=ids_slice, chunks=chunks)
            lon = self.pmesh_prep("Longitude", ids=ids_slice, chunks=chunks)

            nonan = ~da.isnan(x)
            flat_x = x[nonan].compute()
            flat_lat = lat[nonan].compute()
            flat_lon = lon[nonan].compute()

            x_grid = griddata(
                (flat_lon, flat_lat),
                flat_x,
                (lon_grid, lat_grid),
                method=method,
                rescale=True,
            )

            cloud_points = _ndim_coords_from_arrays((flat_lon, flat_lat))
            regrid_points = _ndim_coords_from_arrays((lon_grid.ravel(), lat_grid.ravel()))
            tri = Delaunay(cloud_points)

            outside_hull = np.zeros(lon_grid.size).astype(bool)
            if method == "nearest":
                # filter out the extrapolated points when using nearest neighbors
                outside_hull = tri.find_simplex(regrid_points) < 0

            # filter out points that fall in large triangles
            # create a new scipy.spatial.Delaunay object with only the large triangles
            large_triangles = ~filter_large_triangles(cloud_points, tri)
            large_triangle_ids = np.where(large_triangles)[0]
            subset_tri = tri  # this doesn't preserve tri, effectively just a renaming
            # the find_simplex method only needs the simplices and neighbors
            subset_tri.nsimplex = large_triangle_ids.size
            subset_tri.simplices = tri.simplices[large_triangles]
            subset_tri.neighbors = tri.neighbors[large_triangles]
            # update neighbors
            for i, triangle in enumerate(subset_tri.neighbors):
                for j, neighbor_id in enumerate(triangle):
                    if neighbor_id in large_triangle_ids:
                        # reindex the neighbors to match the size of the subset
                        subset_tri.neighbors[i, j] = np.where(large_triangle_ids == neighbor_id)[0]
                    elif neighbor_id >= 0 and (neighbor_id not in large_triangle_ids):
                        # that neighbor was a "normal" triangle that should not exist in the subset
                        subset_tri.neighbors[i, j] = -1
            inside_large_triangles = subset_tri.find_simplex(regrid_points, bruteforce=True) >= 0
            invalid_slice = np.logical_or(outside_hull, inside_large_triangles)
            x_grid[invalid_slice.reshape(x_grid.shape)] = np.nan

            x_grid_list.append(x_grid)

        stacked_grid = da.stack(x_grid_list, axis=0)
        x_grid_avg = da.nanmean(stacked_grid, axis=0)

        return lon_grid, lat_grid, x_grid_avg

    @timeit
    def heatmap(
        self,
        var: str,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        vminmax: Optional[Annotated[Sequence[float], 2]] = None,
        latlon: bool = False,
        ratio: bool = False,
        ylim: Optional[Annotated[Sequence[float], 2]] = None,
        save_path: Optional[str] = None,
        extra_id: Optional[int] = None,
        ids: Optional[List[int]] = None,
        option: Optional[str] = None,
        option_axis: int = 2,
        chunks: Union[str, Tuple] = "auto",
        lon_lim: Optional[Annotated[Sequence[float], 2]] = None,
        lat_lim: Optional[Annotated[Sequence[float], 2]] = None,
        n: Optional[int] = None,
        method: str = "cubic",
        save_nc: Optional[Annotated[Sequence[str], 2]] = None,
        ax: plt.Axes = None,
        res: float = 20,
        scale: float = 1.0,
    ) -> None:
        """
        Make a heatmap of the given variable
        var: key contained in the variable to search (uses msat_nc fetch method)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        vminmax: min and max value to be shown with the colorbar
        ratio: if True, plots the variable divided by its median
        ylim: sets the vertical axis range (in cross track pixel indices)
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables and when "option" is None
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std'), for example to plot a heatmap of the maximum radiance
        option_axis: the axis along which the stat is applied (2 is typically along the spectral dimension, remember the variables are transposed when read)
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        lon_lim: [min,max] longitudes for the gridding
        lat_lim: [min,max] latitudes for the gridding
        n: number of files chunked together for the gridding
        method: griddata interpolation method, only used if lon_lim and lat_lim are given
        save_nc: [nc_file_path,varname] list containing the full path to the output L3 netcdf file and the name the variable will have in the file
        ax: if given, make the plot in the given matplotlib axes object
        res: the resolution (in meters) of the grid with lon_lim and lat_lim are given
        scale: a factor with which the data will be scaled
        """
        if ids is None:
            ids = self.ids
        if n is None:
            n = len(ids)
        if ax is None:
            self.init_plot(1)
            fig, ax = self.fig, self.ax
            fig.set_size_inches(8, 5)
        else:
            save_path = False  # when an input axis is given, don't try saving a figure

        if ylim:
            ax.set_ylim(*ylim)

        gridded = (lon_lim is not None) and (lat_lim is not None)

        if not gridded:
            x = self.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                ids=ids,
                option=option,
                option_axis=option_axis,
                ratio=ratio,
                chunks=chunks,
            )
            x = x * scale
            if latlon:
                if self.is_l2:
                    grp = "Level1"
                else:
                    grp = "Geolocation"
                lat = self.pmesh_prep(
                    "Latitude",
                    grp=grp,
                    sv_var=sv_var,
                    extra_id=extra_id,
                    ids=ids,
                    ratio=ratio,
                    chunks=chunks,
                )
                lon = self.pmesh_prep(
                    "Longitude",
                    grp=grp,
                    sv_var=sv_var,
                    extra_id=extra_id,
                    ids=ids,
                    ratio=ratio,
                    chunks=chunks,
                )
        if gridded and self.use_dask:
            lon_grid, lat_grid, gridded_x = self.grid_prep(
                var,
                lon_lim,
                lat_lim,
                n=n,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                ids=ids,
                option=option,
                option_axis=option_axis,
                ratio=ratio,
                chunks=chunks,
                method=method,
                res=res,
            )
            gridded_x = gridded_x * scale

            if save_nc:
                with ncdf.Dataset(save_nc[0], "r+") as outfile:
                    if "atrack" not in save_nc.dimensions:
                        outfile.createDimension("atrack", lat_grid.shape([1]))
                    if "xtrack" not in outfile.dimensions:
                        outfile.createDimension("atrack", lat_grid.shape([0]))
                    if "latitude" not in outfile.variables:
                        outfile.createVariable("latitude", lat_grid.shape, ("xtrack", "atrack"))
                        outfile["latitude"][:] = lat_grid
                    if "longitude" not in outfile.variables:
                        outfile.createVariable("longitude", lat_grid.shape, ("xtrack", "atrack"))
                        outfile["longitude"][:] = lon_grid
                    if save_nc[1] not in outfile.variables:
                        outfile.createVariable(save_nc[1], lat_grid.shape, ("xtrack", "atrack"))
                    outfile[save_nc[1]] = gridded_x

            if vminmax is None:
                m = ax.pcolormesh(lon_grid, lat_grid, gridded_x, cmap="viridis")
            else:
                m = ax.pcolormesh(
                    lon_grid,
                    lat_grid,
                    gridded_x,
                    cmap="viridis",
                    vmin=vminmax[0],
                    vmax=vminmax[1],
                )
        else:
            if gridded:
                print("/!\\ the gridded argument only works when self.use_dask is True")
            if vminmax is None:
                if latlon:
                    m = ax.pcolormesh(
                        lon[self.valid_xtrack],
                        lat[self.valid_xtrack],
                        x[self.valid_xtrack],
                        cmap="viridis",
                    )
                else:
                    m = ax.pcolormesh(x, cmap="viridis")
            else:
                if latlon:
                    m = ax.pcolormesh(
                        lon[self.valid_xtrack],
                        lat[self.valid_xtrack],
                        x[self.valid_xtrack],
                        cmap="viridis",
                        vmin=vminmax[0],
                        vmax=vminmax[1],
                    )
                else:
                    m = ax.pcolormesh(x, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])

        if var == "dp":
            lab = "$\Delta P$"
        elif sv_var:
            lab = sv_var
        else:
            lab = var

        units = self.fetch_units(var)
        if units:
            lab = f"{lab} ({units})"

        if option:
            lab = f"{option} {lab}"

        plt.colorbar(m, label=lab, ax=ax)

        if self.dates is not None:
            dates = sorted([self.dates[i] for i in ids])
            ax.set_title(
                f"{datetime.strftime(dates[0],'%Y%m%dT%H%M%S')} to {datetime.strftime(dates[-1],'%Y%m%dT%H%M%S')}"
            )
        if gridded or latlon:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            ax.set_xlabel("along-track index")
            ax.set_ylabel("cross-track index")

        # disable scientific notations in axis numbers
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)

        if save_path:
            fig.savefig(save_path)
            # also save a pickled version to be able to reopen the interactive matplotlib figure
            with open(f"{save_path}.pickle", "wb") as outfile:
                pickle.dump(fig, outfile)

    def search(self, key: str) -> None:
        """
        Use the msat_nc.search method on one of the files
        key: the string you would like to search for (included in groups or variables)
        """

        self.msat_files[self.ids[0]].search(key)

    def fetch(
        self, key: str, msat_id: int, chunks: Union[str, Tuple] = "auto"
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Use the msat_nc.fetch method on the file corresponding to the given id (from self.ids)
        key: the string you would like to search for (included in groups or variables)
        msat_id: id of the msat_file ()
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        return self.msat_files[self.ids[msat_id]].fetch(key, chunks=chunks)

    def fetch_units(self, key: str) -> str:
        """
        Use the msat_mc.fetch_units method on the first file in the list to return the units of the variable that first matches key
        key: the string you would like to search for (included in groups or variables)
        """
        return self.msat_files[self.ids[0]].fetch_units(key)

    def fetch_varpath(self, key, grp: Optional[str] = None) -> str:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        grp: if given, searches in the given group only
        """
        return self.msat_files[self.ids[0]].fetch_varpath(key, grp=grp)

    def show_all(self) -> None:
        """
        Display all the groups, variables (+dimensions) in the netcdf files
        """
        self.msat_files[self.ids[0]].show_all()

    def show_group(self, grp: str) -> None:
        """
        Show all the variable names and dimensions of a given group
        grp: complete group name
        """
        self.msat_files[self.ids[0]].show_group(grp)

    def get_sv_slice(self, var: str) -> np.ndarray:
        """
        Get the state vector index for the given variable
        var: complete state vector variable name
        """
        return self.msat_files[self.ids[0]].get_sv_slice(var)

    def show_sv(self) -> None:
        """
        Display the state vector variable names
        """
        self.msat_files[self.ids[0]].show_sv()

    def set_use_dask(self, use_dask: bool) -> None:
        self.use_dask = use_dask
        if hasattr(self, "msat_files"):
            for msat_file in self.msat_files.values():
                msat_file.use_dask = use_dask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="full path to MethaneAIR file")
    args = parser.parse_args()

    msat = msat_file(args.infile)

    return msat


if __name__ == "__main__":
    msat = main()
