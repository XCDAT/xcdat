import xarray as xr
from xarray.tutorial import load_dataset as xr_load_dataset

import xcdat.bounds  # noqa: F401


def load_dataset(name: str) -> xr.Dataset:
    ds = xr_load_dataset(name)
    ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "T"])

    return ds
