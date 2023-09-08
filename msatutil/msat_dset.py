from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.client import Client
from netCDF4 import Dataset
from typing import Union


class msat_dset(Dataset):
    """
    Class to open a netcdf file on a google cloud bucket
    Download the file in memory and produce a netCDF4._netCDF4.Dataset object
    """

    def __init__(self, nc_target: Union[str, Blob], client: Client = None):
        """
        nc_target can be:
            - a netcdf file path (str)
            - a google cloud file path starting with gs:// (str)
            - a google cloud blob (Blob)

        client (Client): used when nc_target is a blob, defaults to storage.Client()
        """
        if "blob" in str(type(nc_target)).lower() or nc_target.startswith("gs://"):
            if nc_target.startswith("gs://"):
                if client is None:
                    client = storage.Client()
                nc_target = Blob.from_string(nc_target, client=client)
            with nc_target.open("rb") as gcloud_file:
                data = gcloud_file.read()
            filename = "memory"
        else:
            data = None
            filename = nc_target

        super().__init__(filename, memory=data)
