from __future__ import annotations
import os
import numpy as np
import netCDF4 as ncdf
from typing import Optional, Tuple, Union, List, Dict
import dask.array as da


class MSATError(Exception):
    pass


class msat_nc:
    """
    This class holds a netCDF.Dataset for a MethaneSAT/AIR L1B or L2 file.
    It contains methods that help navigate the dataset more quickly
    """

    def __init__(self, infile: str, use_dask: bool = False) -> None:
        self.use_dask = use_dask
        self.exists = os.path.exists(infile)
        if not self.exists:
            raise MSATError(f"Wrong path: {infile}")
        else:
            self.nc_dset = ncdf.Dataset(infile, "r")

        self.dp = None
        self.datetimes = None
        self.is_postproc = "product" in self.nc_dset.groups
        self.is_l2_met = "Surface_Band1" in self.nc_dset.groups
        self.is_l2 = not self.is_l2_met and (("Level1" in self.nc_dset.groups) or self.is_postproc)
        self.is_l1 = True not in [self.is_l2, self.is_l2_met, self.is_postproc]
        self.varpath_list = None

        # dictionary that maps all the dimensions names across L1/L2 file versions to a common set of names
        # use the get_dim_map method to get a mapping of a given variables dimensions that use the dimensions names from the common set
        # common set: ["one","xtrack","atrack","xtrack_edge","atrack_edge","lev","lev_edge","corner","spectral_channel","xmx","nsubx"]
        self.dim_name_map = {
            "one": "one",
            "o": "one",
            "imx": "xtrack",
            "xtrack": "xtrack",
            "across_track": "xtrack",
            "x": "xtrack",
            "imx_e": "xtrack_edge",
            "xtrack_edge": "xtrack_edge",
            "jmx": "atrack",
            "atrack": "atrack",
            "along_track": "atrack",
            "y": "atrack",
            "jmx_e": "atrack_edge",
            "atrack_edge": "atrack_edge",
            "lmx": "lev",
            "lev": "lev",
            "level": "lev",
            "levels": "lev",
            "z": "lev",
            "lmx_e": "lev_edge",
            "lev_edge": "lev_edge",
            "ze": "lev_edge",
            "vertices": "corner",
            "four": "corner",
            "c": "corner",
            "corner": "corner",
            "nv": "corner",
            "w1": "spectral_channel",
            "wmx_1": "spectral_channel",
            "spectral_channel": "spectral_channel",
            "xmx": "xmx",
            "nsubx": "nsubx",
            "p1": "one",
            "p2": "two",
            "p3": "three",
            "w1_alb": "alb_wvl",
            "k1_alb": "alb_kernel",
            "p1_alb": "alb_poly",
        }

        self.dim_size_map = {
            self.dim_name_map[dim]: dim_item.size
            for dim, dim_item in self.nc_dset.dimensions.items()
        }

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.nc_dset.close()

    def read_dp(self) -> None:
        """
        Getting the retrieved minus prior surface pressure in self.dp
        """
        try:  # this is a try so we can use the same class to read the L1 files
            if "o2dp_fit_diagnostics" in self.nc_dset.groups:
                self.dp = self.nc_dset["o2dp_fit_diagnostics"]["bias_corrected_delta_pressure"][:]
                return
            else:
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
        except Exception:
            pass

    def get_var(
        self,
        var: str,
        grp: Optional[str] = None,
        chunks: Union[str, Tuple] = "auto",
    ) -> Union[np.ndarray, np.ma.masked_array, da.core.Array]:
        """
        return a variable array from the netcdf file
        var: complete variable name
        grp: complete group name
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        if var.lower() == "dp":
            return self.dp
        elif var.lower() in ["datetime", "datetimes"]:
            return self.datetimes
        elif grp is not None:
            if self.use_dask:
                return da.from_array(self.nc_dset[grp][var], chunks=chunks)
            return self.nc_dset[grp][var][tuple()]
        else:
            if self.use_dask:
                return da.from_array(self.nc_dset[var], chunks=chunks)
            return self.nc_dset[var][tuple()]

    def get_units(self, var: str, grp="") -> str:
        """
        get the units of the given variable
        var: complete variable name
        grp: complete group name
        """
        if grp:
            nc_var = self.nc_dset[grp][var]
        else:
            nc_var = self.nc_dset[var]

        units = ""
        if hasattr(nc_var, "units"):
            units = nc_var.units

        return units

    def show_var(self, var: str, grp: Optional[str] = None) -> None:
        """
        display the given variable metadata
        var: complete variable name
        grp: complete group name
        """
        if grp:
            print(self.nc_dset[grp][var])
        else:
            print(self.nc_dset[var])

    def show_all(self) -> None:
        """
        Show all the groups names, variable names, and variable dimensions
        """
        if self.nc_dset.groups:
            for grp in self.nc_dset.groups:
                print(grp)
                for var in self.nc_dset[grp].variables:
                    print("\t", var, self.nc_dset[grp][var].dimensions)
        elif self.nc_dset.variables:
            for var in self.nc_dset.variables:
                print(var, self.nc_dset[var].dimensions)

    def show_group(self, grp: str) -> None:
        """
        Show all the variable names and dimensions of a given group
        grp: complete group name
        """
        if self.nc_dset.groups:
            for var in self.nc_dset[grp].variables:
                print(var, self.nc_dset[grp][var].dimensions)

    def show_sv(self) -> None:
        """
        Display the state vector variable names
        """
        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__
        for key, val in sv_dict.items():
            if type(val) == str:
                val = val.strip()
            print(f"{key.strip()}: {val}")

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

    def search(self, key: str) -> None:
        """
        print out groups and variables that include the key (all lowercase checks)
        key: the string you would like to search for (included in groups or variables)
        """
        key = key.lower()
        if self.nc_dset.variables:
            for var in self.nc_dset.variables:
                if key in var.lower():
                    print(f"VAR: {var} {self.nc_dset[var].dimensions}")
        if self.nc_dset.groups:
            for grp in self.nc_dset.groups:
                if key in grp.lower():
                    print(f"GROUP: {grp}")
                for var in self.nc_dset[grp].variables:
                    if key in var.lower():
                        print(f"GROUP: {grp}\tVAR: {var} {self.nc_dset[grp][var].dimensions}")
                    if var == "APosterioriState":
                        sv_dict = self.nc_dset["SpecFitDiagnostics/APosterioriState"].__dict__
                        for sv_key, val in sv_dict.items():
                            if sv_key.startswith("SubStateName") and key in val.strip().lower():
                                print(
                                    f"GROUP: {grp}\tVAR: {var} {self.nc_dset[grp][var].dimensions} \tSV_VAR: {val.strip()} \tSV_SLICE: {list(self.get_sv_slice(val.strip()))}"
                                )

    def fetch(
        self, key: str, chunks: Union[str, Tuple] = "auto"
    ) -> Union[np.ndarray, np.ma.masked_array, da.core.Array]:
        """
        retrieves the first variable that matches key (all lowercase checks)
        key: the string you would like to search for (included in groups or variables)
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        key = key.lower()
        if key == "dp":
            return self.dp
        elif key in ["datetime", "datetimes"]:
            return self.datetimes
        if self.nc_dset.variables:
            for var in self.nc_dset.variables:
                if key in var.lower():
                    if self.use_dask:
                        return da.from_array(self.nc_dset[var], chunks=chunks)
                    return self.nc_dset[var][:]
        if self.nc_dset.groups:
            for grp in self.nc_dset.groups:
                for var in self.nc_dset[grp].variables:
                    if key in var.lower():
                        if self.use_dask:
                            return da.from_array(self.nc_dset[grp][var], chunks=chunks)
                        return self.nc_dset[grp][var][:]

    def fetch_units(self, key: str) -> str:
        """
        Get the units of the variable that matches key
        key: the string you would like to search for (included in groups or variables)
        """
        key = key.lower()
        units = ""
        if key == "dp":
            units = "hPa"
        elif key in ["datetime", "datetimes"]:
            units = "hours since 1985-01-01"
        else:
            if self.nc_dset.variables:
                for var in self.nc_dset.variables:
                    if key in var.lower():
                        if hasattr(self.nc_dset[var], "units"):
                            units = self.nc_dset[var].units
                            break
                        elif hasattr(self.nc_dset[var], "unit"):
                            units = self.nc_dset[var].unit
                            break
            if self.nc_dset.groups:
                for grp in self.nc_dset.groups:
                    for var in self.nc_dset[grp].variables:
                        if key in var.lower():
                            if hasattr(self.nc_dset[grp][var], "units"):
                                units = self.nc_dset[grp][var].units
                                break
                            elif hasattr(self.nc_dset[grp][var], "unit"):
                                units = self.nc_dset[grp][var].unit
                                break
                    else:
                        continue
                    break

        return units

    def fetch_varpath(self, key: str, grp: Optional[str] = None) -> Union[str, None]:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        grp: if given, searches in the given group only
        """
        key = key.lower()
        if key == "dp":
            return "dp"
        if grp is not None:
            for var in self.nc_dset[grp].variables:
                if key in var.lower():
                    return f"{grp}/{var}"
        if self.nc_dset.variables:
            for var in self.nc_dset.variables:
                if key in var.lower():
                    return var
        if self.nc_dset.groups:
            for grp in self.nc_dset.groups:
                for var in self.nc_dset[grp].variables:
                    if key in var.lower():
                        return f"{grp}/{var}"

    @staticmethod
    def nctime_to_pytime(
        nc_time_var: ncdf.Variable,
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Convert time in a netCDF variable to an array of Python datetime objects.
        """
        if hasattr(nc_time_var, "calendar"):
            calendar = nc_time_var.calendar
        else:
            calendar = "standard"
        if not hasattr(nc_time_var, "units"):
            print(
                "Time variable has no units for num2date conversion, assume hours since 1985-01-01"
            )
            units = "hours since 1985-01-01"
        else:
            units = nc_time_var.units

        return ncdf.num2date(
            nc_time_var[:],
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
        )

    def get_dim_map(self, var_path: str) -> Dict[str, int]:
        """
        For a given key, use fetch_varpath to inspect the corresponding variable and
        return a map of its dimension axes {'dim_name':dim_axis}
        """
        if var_path.lower() == "dp":
            if len(self.dp.shape) == 3:
                var_dim_map = {"one": 0, "atrack": 1, "xtrack": 2}
            else:
                var_dim_map = {"atrack": 0, "xtrack": 1}
        else:
            var_dims = self.nc_dset[var_path].dimensions
            var_dim_map = {self.dim_name_map[dim]: var_dims.index(dim) for dim in var_dims}

        return var_dim_map

    def get_valid_xtrack(self) -> slice:
        """
        Get the valid cross track indices
        """
        if self.is_l2_met:
            return slice(None)

        if self.is_postproc:
            longitude_varpath = "geolocation/longitude"
        elif self.is_l2:
            longitude_varpath = "Level1/Longitude"

        if self.is_l2:
            var_dim_map = self.get_dim_map(longitude_varpath)
            atrack_axis = var_dim_map["atrack"]
            valid_xtrack = np.where(
                ~np.isnan(
                    np.nanmedian(self.nc_dset[longitude_varpath][:], axis=atrack_axis).squeeze()
                )
            )[0]
        else:
            var_dim_map = self.get_dim_map("Band1/Radiance")
            spec_axis = var_dim_map["spectral_channel"]
            atrack_axis = var_dim_map["atrack"]
            xtrack_axis = var_dim_map["xtrack"]
            rad = self.nc_dset["Band1/Radiance"][:]
            rad = rad.transpose(atrack_axis, xtrack_axis, spec_axis)
            valid_xtrack = np.where(np.nanmedian(np.nansum(rad, axis=2), axis=0) > 0)[0]
        if len(valid_xtrack) == 0:
            print(self.nc_dset.filepath(), " has no valid xtrack")
            valid_xtrack_slice = slice(None)
        else:
            valid_xtrack_slice = slice(valid_xtrack[0], valid_xtrack[-1] + 1)

        return valid_xtrack_slice

    def get_valid_rad(self) -> Union[slice, None]:
        """
        Get the valid radiance indices
        """
        if self.is_postproc:
            return None
        if self.is_l2 and self.has_var("RTM_Band1/Radiance_I"):
            rad_var_path = "RTM_Band1/Radiance_I"
        elif self.is_l2:
            rad_var_path = None
        elif not self.is_l2:
            rad_var_path = "Band1/Radiance"

        if rad_var_path is None:
            valid_rad_slice = None
        else:
            var_dim_map = self.get_dim_map(rad_var_path)
            xtrack_axis = var_dim_map["xtrack"]
            atrack_axis = var_dim_map["atrack"]
            valid_rad = np.where(
                np.nansum(self.nc_dset[rad_var_path][:], axis=(xtrack_axis, atrack_axis)) > 0
            )[0]

            if valid_rad.size != 0:
                valid_rad_slice = slice(valid_rad[0], valid_rad[-1] + 1)
            else:
                print(self.nc_dset.filepath(), " has no valid radiances")
                valid_rad_slice = None

        return valid_rad_slice

    def get_var_paths(self) -> List[str]:
        """
        Get a list of all the full variable paths in the netcdf file
        """

        if self.varpath_list is not None:
            return self.varpath_list

        varpath_list = []

        for v in self.nc_dset.variables:
            varpath_list += [v]
        for g in self.nc_dset.groups:
            for v in self.nc_dset[g].variables:
                varpath_list += [f"{g}/{v}"]

        self.varpath_list = varpath_list

        return varpath_list

    def has_var(self, var) -> bool:
        """
        Check if the netcdf file has the given variable
        """
        varpath_list = self.get_var_paths()
        return var in varpath_list
