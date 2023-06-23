from google.cloud import storage
from google.cloud.storage.blob import Blob
from netCDF4 import Dataset


class cloud_dset(Dataset):
    """
    Class to open a netcdf file on a google cloud bucket
    Download the file in memory and produce a netCDF4._netCDF4.Dataset object
    """
    def __init__(self, cloud_file_path, client=None):
        if client is None:
            client = storage.Client()
        blob = Blob.from_string(cloud_file_path, client=client)
        with blob.open("rb") as infile:
            data = infile.read()
        super().__init__("memory", memory=data)
