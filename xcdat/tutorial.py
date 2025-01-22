from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr
from xarray.tutorial import open_dataset as xr_open_dataset

import xcdat.bounds  # noqa: F401

if TYPE_CHECKING:
    import os

    from xarray.backends.api import T_Engine


def open_dataset(
    name: str,
    cache: bool = True,
    cache_dir: None | str | os.PathLike = None,
    *,
    engine: T_Engine = None,
    **kws,
) -> xr.Dataset:
    """Open a dataset from the online repository (requires internet).

    This function is a wrapper around ``xarray.tutorial.open_dataset`` that
    adds missing bounds to the dataset. If a local copy of the dataset file is
    found then always use that to avoid network traffic.

    Available datasets:

    * ``"air_temperature"``: NCEP reanalysis subset
    * ``"air_temperature_gradient"``: NCEP reanalysis subset with approximate x,y gradients
    * ``"basin_mask"``: Dataset with ocean basins marked using integers
    * ``"ASE_ice_velocity"``: MEaSUREs InSAR-Based Ice Velocity of the Amundsen Sea Embayment, Antarctica, Version 1
    * ``"rasm"``: Output of the Regional Arctic System Model (RASM)
    * ``"ROMS_example"``: Regional Ocean Model System (ROMS) output
    * ``"tiny"``: small synthetic dataset with a 1D data variable
    * ``"era5-2mt-2019-03-uk.grib"``: ERA5 temperature data over the UK
    * ``"eraint_uvz"``: data from ERA-Interim reanalysis, monthly averages of upper level data
    * ``"ersstv5"``: NOAA's Extended Reconstructed Sea Surface Temperature monthly averages

    Parameters
    ----------
    name : str
        Name of the file containing the dataset.
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    **kws : dict, optional
        Passed to xarray.open_dataset
    """
    ds = xr_open_dataset(
        name=name, cache=cache, cache_dir=cache_dir, engine=engine, **kws
    )
    ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "T"])

    return ds
