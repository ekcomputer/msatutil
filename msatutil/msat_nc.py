from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as ncdf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Sequence, Tuple, Union, Annotated, List
from collections import OrderedDict
from io import StringIO
import argparse
import dask
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
        self.datetimes = None

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.nc_dset.close()

    def get_var(
        self, var: str, grp: Optional[str] = None, chunks: Union[str, Tuple] = "auto",
    ) -> Union[np.ndarray, np.ma.masked_array, da.core.Array]:
        """
        return a variable array from the netcdf file
        var: complete variable name
        grp: complete group name
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        if var.lower() in ["datetime", "datetimes"]:
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

    def get_sv_slice(self, var: str) -> np.ndarray:
        """
        Get the state vector index for the given variable
        var: complete state vector variable name
        """
        if self.nc_dset.groups and ("SpecFitDiagnostics" not in self.nc_dset.groups):
            print('get_sv_slice needs a msat file with a "SpecFitDiagnostics" group')
            return None
        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__

        for key, val in sv_dict.items():
            if key.startswith("SubStateName") and val.strip() == var:
                num = int(key.split("_")[-1]) - 1
                slice = np.arange(
                    sv_dict["SubState_Idx0"][num] - 1, sv_dict["SubState_Idxf"][num]
                )
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
                        print(
                            f"GROUP: {grp}\tVAR: {var} {self.nc_dset[grp][var].dimensions}"
                        )
                    if var == "APosterioriState":
                        sv_dict = self.nc_dset[
                            "SpecFitDiagnostics/APosterioriState"
                        ].__dict__
                        for sv_key, val in sv_dict.items():
                            if (
                                sv_key.startswith("SubStateName")
                                and key in val.strip().lower()
                            ):
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

    def fetch_varpath(self, key: str, grp: Optional[str] = None) -> str:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        grp: if given, searches in the given group only
        """
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

