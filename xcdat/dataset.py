"""Dataset module for functions related to an xarray.Dataset."""

from __future__ import annotations

import os
import pathlib
from datetime import datetime
from functools import partial
from io import BufferedIOBase
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from dateutil import parser
from dateutil import relativedelta as rd
from xarray.backends.common import AbstractDataStore
from xarray.coding.cftime_offsets import get_date_type
from xarray.coding.times import convert_times, decode_cf_datetime
from xarray.coding.variables import lazy_elemwise_func, pop_to, unpack_for_decoding
from xarray.core.types import NestedSequence
from xarray.core.variable import as_variable

from xcdat import bounds as bounds_accessor  # noqa: F401
from xcdat._logger import _setup_custom_logger
from xcdat.axis import CFAxisKey, _get_all_coord_keys, swap_lon_axis
from xcdat.axis import center_times as center_times_func

logger = _setup_custom_logger(__name__)

#: List of non-CF compliant time units.
NON_CF_TIME_UNITS: List[str] = ["month", "months", "year", "years"]

# Type annotation for the `paths` arg.
Paths = Union[
    str,
    pathlib.Path,
    List[str],
    List[pathlib.Path],
    List[List[str]],
    List[List[pathlib.Path]],
]


def open_dataset(
    path: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    data_var: Optional[str] = None,
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None = ("X", "Y"),
    decode_times: bool = True,
    center_times: bool = False,
    lon_orient: Optional[Tuple[float, float]] = None,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wraps ``xarray.open_dataset()`` with post-processing options.

    Parameters
    ----------
    path : str, Path, file-like or DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). Byte-strings or file-like
        objects are opened by scipy.io.netcdf (netCDF3) or h5py (netCDF4/HDF).
    data_var: Optional[str], optional
        The key of the non-bounds data variable to keep in the Dataset,
        alongside any existing bounds data variables, by default None.
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None
        List of CF axes to try to add bounds for (if missing), by default
        ("X", "Y"). Set to None to not add any missing bounds. Please note that
        bounds are required for many xCDAT features.

        * This parameter calls :py:func:`xarray.Dataset.bounds.add_missing_bounds`
        * Supported CF axes include "X", "Y", "Z", and "T"
        * By default, missing "T" bounds are generated using the time frequency
          of the coordinates. If desired, refer to
          :py:func:`xarray.Dataset.bounds.add_time_bounds` if you require more
          granular configuration for how "T" bounds are generated.
    decode_times: bool, optional
        If True, attempt to decode times encoded in the standard NetCDF
        datetime format into cftime.datetime objects. Otherwise, leave them
        encoded as numbers. This keyword may not be supported by all the
        backends, by default True.
    center_times: bool, optional
        If True, attempt to center time coordinates using the midpoint between
        its upper and lower bounds. Otherwise, use the provided time
        coordinates, by default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists).
        Either `(-180, 180)` or `(0, 360)`, by default None. Supported options
        include:

        * None:  use the current orientation (if the longitude axis exists)
        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation
    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``. Refer to the
        [1]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_dataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------
    .. [1] https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html
    """
    ds = xr.open_dataset(path, decode_times=False, **kwargs)  # type: ignore

    if decode_times:
        try:
            ds = decode_time(ds)
        except KeyError as err:
            logger.warning(err)

    ds = _postprocess_dataset(ds, data_var, center_times, add_bounds, lon_orient)

    return ds


def open_mfdataset(
    paths: str | NestedSequence[str | os.PathLike],
    data_var: Optional[str] = None,
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None = ("X", "Y"),
    decode_times: bool = True,
    center_times: bool = False,
    lon_orient: Optional[Tuple[float, float]] = None,
    data_vars: Literal["minimal", "different", "all"] | List[str] = "minimal",
    preprocess: Optional[Callable] = None,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wraps ``xarray.open_mfdataset()`` with post-processing options.

    Parameters
    ----------
    paths : str | NestedSequence[str | os.PathLike]
        Paths to dataset files. Paths can be given as strings or as pathlib.Path
        objects. Supported options include:

        * Directory path (e.g., ``"path/to/files"``), which is converted
          to a string glob of `*.nc` files
        * String glob (e.g., ``"path/to/files/*.nc"``), which is expanded
          to a 1-dimensional list of file paths
        * File path to dataset (e.g., ``"path/to/files/file1.nc"``)
        * List of file paths (e.g., ``["path/to/files/file1.nc", ...]``).
          If concatenation along more than one dimension is desired, then
          ``paths`` must be a nested list-of-lists (see [2]_
          ``xarray.combine_nested`` for details).
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None
        List of CF axes to try to add bounds for (if missing), by default
        ("X", "Y"). Set to None to not add any missing bounds. Please note that
        bounds are required for many xCDAT features.

        * This parameter calls :py:func:`xarray.Dataset.bounds.add_missing_bounds`
        * Supported CF axes include "X", "Y", "Z", and "T"
        * By default, missing "T" bounds are generated using the time frequency
          of the coordinates. If desired, refer to
          :py:func:`xarray.Dataset.bounds.add_time_bounds` if you require more
          granular configuration for how "T" bounds are generated.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    decode_times: bool, optional
        If True, attempt to decode times encoded in the standard NetCDF
        datetime format into cftime.datetime objects. Otherwise, leave them
        encoded as numbers. This keyword may not be supported by all the
        backends, by default True.
    center_times: bool, optional
        If True, attempt to center time coordinates using the midpoint between
        its upper and lower bounds. Otherwise, use the provided time
        coordinates, by default False.
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists),
        by default None. Supported options include:

        * None:  use the current orientation (if the longitude axis exists)
        * (-180, 180): represents [-180, 180) in math notation
        * (0, 360): represents [0, 360) in math notation
    data_vars: {"minimal", "different", "all" or list of str}, optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included, the default value.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.

        The ``data_vars`` kwarg defaults to ``"minimal"``, which concatenates
        data variables in a manner where only data variables in which the
        dimension already appears are included. For example, the time dimension
        will not be concatenated to the dimensions of non-time data variables
        such as "lat_bnds" or "lon_bnds". ``data_vars="minimal"`` is required for
        some xCDAT functions, including spatial averaging where a reduction is
        performed using the lat/lon bounds.
    preprocess : Optional[Callable], optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding["source"]``.
    **kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to
        the [3]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Notes
    -----
    ``xarray.open_mfdataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------
    .. [2] https://docs.xarray.dev/en/stable/generated/xarray.combine_nested.html
    .. [3] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html
    """
    if isinstance(paths, str) or isinstance(paths, pathlib.Path):
        if os.path.isdir(paths):
            paths = _parse_dir_for_nc_glob(paths)

    preprocess = partial(_preprocess, decode_times=decode_times, callable=preprocess)

    ds = xr.open_mfdataset(
        paths,
        decode_times=False,
        data_vars=data_vars,
        preprocess=preprocess,
        **kwargs,  # type: ignore
    )

    ds = _postprocess_dataset(ds, data_var, center_times, add_bounds, lon_orient)

    return ds


def decode_time(dataset: xr.Dataset) -> xr.Dataset:
    """Decodes CF and non-CF time coordinates and time bounds using ``cftime``.

    By default, ``xarray`` only supports decoding time with CF compliant units
    [5]_. This function enables also decoding time with non-CF compliant units.
    It skips decoding time coordinates that have already been decoded as
    ``"datetime64[ns]"`` or ``cftime.datetime``.

    For time coordinates to be decodable, they must have a "calendar" attribute
    set to a CF calendar type supported by ``cftime``. CF calendar types
    include "noleap", "360_day", "365_day", "366_day", "gregorian",
    "proleptic_gregorian", "julian", "all_leap", or "standard". They must also
    have a "units" attribute set to a format supported by xCDAT ("months since
    ..." or "years since ...").

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with numerically encoded time coordinates and time bounds (if
        they exist). If the time coordinates cannot be decoded then the original
        dataset is returned.

    Returns
    -------
    xr.Dataset
        Dataset with decoded time coordinates and time bounds (if they exist) as
        ``cftime`` objects.

    Raises
    ------
    KeyError
        If time coordinates were not detected in the dataset, either because they
        don't exist at all or their CF attributes (e.g., 'axis' or
        'standard_name') are not set.

    Notes
    -----
    Time coordinates are represented by ``cftime.datetime`` objects because
    it is not restricted by the ``pandas.Timestamp`` range (years 1678 through
    2262). Refer to [6]_ and [7]_ for more information on this limitation.

    References
    -----
    .. [5] https://cfconventions.org/cf-conventions/cf-conventions.html#time-coordinate
    .. [6] https://docs.xarray.dev/en/stable/user-guide/weather-climate.html#non-standard-calendars-and-dates-outside-the-timestamp-valid-range
    .. [7] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timestamp-limitations

    Examples
    --------

    Decode the time coordinates in a Dataset:

    >>> from xcdat.dataset import decode_time
    >>>
    >>> ds.time
    <xarray.DataArray 'time' (time: 3)>
    array([0, 1, 2])
    Coordinates:
    * time     (time) int64 0 1 2
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time
        calendar:       noleap
    >>>
    >>> ds_decoded = decode_time(ds)
    >>> ds_decoded.time
    <xarray.DataArray 'time' (time: 3)>
    array([cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True),
           cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True),
           cftime.DatetimeNoLeap(1850, 1, 1, 0, 0, 0, 0, has_year_zero=True)],
           dtype='object')
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-01 2001-01-01 2002-01-01
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time
        calendar:       noleap

    View time encoding information:

    >>> ds_decoded.time.encoding
    {'source': None,
     'dtype': dtype('int64'),
     'original_shape': (3,),
     'units': 'years since 2000-01-01',
     'calendar': 'noleap'}
    """
    ds = dataset.copy()
    coord_keys = _get_all_coord_keys(ds, "T")

    if len(coord_keys) == 0:
        raise KeyError(
            "No time coordinates were found in this dataset to decode. If time "
            "coordinates were expected to exist, make sure they are detectable by "
            "setting the CF 'axis' or 'standard_name' attribute (e.g., "
            "ds['time'].attrs['axis'] = 'T' or "
            "ds['time'].attrs['standard_name'] = 'time'). Afterwards, try decoding "
            "again with `xcdat.decode_time`."
        )

    for key in coord_keys:
        coords = ds[key].copy()

        if _is_decodable(coords) and not _is_decoded(coords):
            if coords.attrs.get("calendar") is None:
                coords.attrs["calendar"] = "standard"
                logger.warning(
                    f"'{coords.name}' does not have a calendar attribute set. "
                    "Defaulting to CF 'standard' calendar."
                )

            decoded_time = _decode_time(coords)
            ds = ds.assign_coords({coords.name: decoded_time})

            try:
                bounds = ds.bounds.get_bounds("T", var_key=coords.name)
            except KeyError:
                bounds = None

            if bounds is not None and not _is_decoded(bounds):
                # Bounds don't typically store the "units" and "calendar"
                # attributes required for decoding, so these attributes need to be
                # copied from the coordinates.
                bounds.attrs.update(
                    {
                        "units": coords.attrs["units"],
                        "calendar": coords.attrs["calendar"],
                    }
                )
                decoded_bounds = _decode_time(bounds)
                ds = ds.assign({bounds.name: decoded_bounds})

    return ds


def _parse_dir_for_nc_glob(dir_path: str | pathlib.Path) -> str:
    """Parses a directory for a glob of `*.nc` paths.

    Parameters
    ----------
    dir_path : str | pathlib.Path
        The directory.

    Returns
    -------
    str | pathlib.Path
        A glob of `*.nc` paths in the directory.
    """
    file_list = os.listdir(dir_path)

    if len(file_list) == 0:
        raise ValueError(
            f"The directory {dir_path} has no netcdf (`.nc`) files to open."
        )

    if isinstance(dir_path, pathlib.Path):
        dir_path = str(dir_path)

    return os.path.join(dir_path, "*.nc")


def _preprocess(
    ds: xr.Dataset, decode_times: Optional[bool], callable: Optional[Callable] = None
) -> xr.Dataset:
    """Preprocesses each dataset passed to ``open_mfdataset()``.

    This function accepts a user specified preprocess function, which is
    executed before additional internal preprocessing functions.

    An internal call to ``decode_time()`` is performed, which decodes
    both CF and non-CF time coordinates and bounds (if they exist). By default,
    if ``decode_times=False`` is passed to ``open_mfdataset()``,  xarray will
    concatenate time values using the first dataset's ``units`` attribute. This
    results in an issue for cases where the numerically encoded time values are
    the same and the ``units`` attribute differs between datasets. For example,
    two files have the same time values, but the units of the first file is
    "months since 2000-01-01" and the second is "months since 2001-01-01". Since
    the first dataset's units are used in xarray for concatenating datasets, the
    time values corresponding to the second file will be dropped since they
    appear to be the same as the first file. Calling ``decode_time()``
    on each dataset individually before concatenating solves the aforementioned
    issue.

    Parameters
    ----------
    ds : xr.Dataset
        The Dataset.
    callable : Optional[Callable], optional
        A user specified optional callable function for preprocessing.

    Returns
    -------
    xr.Dataset
        The preprocessed Dataset.
    """
    ds_new = ds.copy()

    if callable:
        ds_new = callable(ds)

    if decode_times:
        try:
            ds_new = decode_time(ds_new)
        except KeyError as err:
            logger.warning(err)

    return ds_new


def _postprocess_dataset(
    dataset: xr.Dataset,
    data_var: Optional[str] = None,
    center_times: bool = False,
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None = ("X", "Y"),
    lon_orient: Optional[Tuple[float, float]] = None,
) -> xr.Dataset:
    """Post-processes a Dataset object.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    center_times: bool, optional
        If True, center time coordinates using the midpoint between its upper
        and lower bounds. Otherwise, use the provided time coordinates, by
        default False.
    add_bounds: List[CFAxisKey] | Tuple[CFAxisKey, ...] | None
        List of CF axes to try to add bounds for (if missing), default
        ("X", "Y"). Set to None to not add any missing bounds.

        * This parameter simply calls :py:func:`xarray.Dataset.bounds.add_missing_bounds`
        * Supported CF axes include "X", "Y", "Z", and "T"
        * Bounds are required for many xCDAT features
        * If desired, use :py:func:`xarray.Dataset.bounds.add_time_bounds`
          if you require more granular configuration for how "T" bounds
          are generated
    lon_orient: Optional[Tuple[float, float]], optional
        The orientation to use for the Dataset's longitude axis (if it exists),
        by default None.

        Supported options:

          * None:  use the current orientation (if the longitude axis exists)
          * (-180, 180): represents [-180, 180) in math notation
          * (0, 360): represents [0, 360) in math notation

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    ValueError
        If ``center_times==True`` but there are no time coordinates.
    ValueError
        If ``lon_orient is not None`` but there are no longitude coordinates.
    """
    ds = dataset.copy()

    if data_var is not None:
        ds = _keep_single_var(dataset, data_var)

    if center_times:
        ds = center_times_func(dataset)

    if add_bounds is not None:
        ds = ds.bounds.add_missing_bounds(axes=add_bounds)

    if lon_orient is not None:
        ds = swap_lon_axis(ds, to=lon_orient, sort_ascending=True)

    return ds


def _is_decodable(coords: xr.DataArray) -> bool:
    """Checks if time coordinates are decodable.

    Time coordinates must have a "units" attribute in a supported format
    to be decodable.

    Parameters
    ----------
    coords : xr.DataArray
        The time coordinates.

    Returns
    -------
    bool
    """
    units = coords.attrs.get("units")

    if units is None:
        logger.warning(
            f"'{coords.name}' does not have a 'units' attribute set so it "
            "could not be decoded. Try setting the 'units' attribute "
            "(`ds.{coords.name}.attrs['units']`) and try decoding again."
        )
        return False

    if isinstance(units, str) and "since" not in units:
        logger.warning(
            f"The 'units' attribute ({units}) for '{coords.name}' is not in the "
            "supported format 'X since Y', so it could not be decoded."
        )
        return False

    return True


def _is_decoded(da: xr.DataArray) -> bool:
    """Check if a time-based DataArray is decoded.

    This is determined by checking if the `encoding` dictionary has "units" and
    "calendar" attributes set.

    Parameters
    ----------
    da : xr.DataArray
        A time-based DataArray (e.g,. coordinates, bounds)

    Returns
    -------
    bool
    """
    units = da.encoding.get("units")
    calendar = da.encoding.get("calendar")

    return calendar is not None and units is not None


def _decode_time(da: xr.DataArray) -> xr.Variable:
    """Lazily decodes a DataArray of numerically encoded time with cftime.

    The ``xr.DataArray`` is converted to an ``xr.Variable`` so that
    ``xr.coding.variables.lazy_elemwise_func`` can be leveraged to lazily decode
    time.

    This function is based on ``xarray.coding.times.CFDatetimeCoder.decode``.

    Parameters
    ----------
    coords : xr.DataArray
        A DataArray of numerically encoded time.

    Returns
    -------
    xr.Variable
        A Variable of time decoded as ``cftime`` objects.
    """
    variable = as_variable(da)
    dims, data, attrs, encoding = unpack_for_decoding(variable)

    units = pop_to(attrs, encoding, "units")
    calendar = pop_to(attrs, encoding, "calendar")

    transform = partial(_get_cftime_coords, units=units, calendar=calendar)
    data = lazy_elemwise_func(data, transform, np.dtype("object"))

    return xr.Variable(dims, data, attrs, encoding)


def _get_cftime_coords(offsets: np.ndarray, units: str, calendar: str) -> np.ndarray:
    """Get an array of cftime coordinates starting from a reference date.

    This function calls xarray's ``decode_cf_datetime()`` if the units are
    CF compliant because ``decode_cf_datetime()`` considers leap days when
    decoding time offsets to ``cftime`` objects.

    For non-CF compliant units ("[months|years] since ..."), this function
    performs custom decoding. It flattens the array, performs decoding on the
    time offsets, then reshapes the array back to its original shape.

    Parameters
    ----------
    offsets : np.ndarray
        An array of numerically encoded time offsets from the reference date.
    units : str
        The time units.
    calendar : str
        The CF calendar type supported by ``cftime``. This includes "noleap",
        "360_day", "365_day", "366_day", "gregorian", "proleptic_gregorian",
        "julian", "all_leap", and "standard".

    Returns
    -------
    np.ndarray
        An array of ``cftime`` coordinates.
    """
    units_type, ref_date = units.split(" since ")

    if units_type not in NON_CF_TIME_UNITS:
        return decode_cf_datetime(offsets, units, calendar=calendar, use_cftime=True)

    offsets = np.asarray(offsets)
    flat_offsets = offsets.ravel()

    # Convert offsets to `np.float64` to avoid "TypeError: unsupported type
    # for timedelta days component: numpy.int64".
    flat_offsets = flat_offsets.astype("float")  # type: ignore

    # We don't need to do calendar arithmetic here because the units and
    # offsets are in "months" or "years", which means leap days should not
    # be factored.
    ref_datetime: datetime = parser.parse(ref_date, default=datetime(2000, 1, 1))
    times = np.array(
        [
            ref_datetime + rd.relativedelta(**{units_type: offset})
            for offset in flat_offsets
        ],
        dtype="object",
    )
    # Convert the array of `datetime` objects into `cftime` objects based on
    # the calendar type.
    date_type = get_date_type(calendar)
    coords = convert_times(times, date_type=date_type)

    # Reshape back to the original shape.
    coords = coords.reshape(offsets.shape)

    return coords


def _keep_single_var(dataset: xr.Dataset, key: str) -> xr.Dataset:
    """Keeps a single non-bounds data variable in the Dataset.

    This function checks if the ``data_var`` key exists in the Dataset and
    it is not related to bounds. If those checks pass, it will subset the
    Dataset to retain that non-bounds ``data_var`` and all bounds data vars.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    key: str
        The key of the non-bounds data variable to keep in the Dataset.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    ValueError
        If the dataset only contains bounds data variables.
    ValueError
        If the specified key does not exist in the dataset.
    ValueError
        If the specified key matches a bounds data variable.
    """
    all_vars = dataset.data_vars.keys()
    bounds_vars = dataset.bounds.keys
    non_bounds_vars = sorted(list(set(all_vars) ^ set(bounds_vars)))

    if len(non_bounds_vars) == 0:
        raise ValueError("This dataset only contains bounds data variables.")

    if key not in all_vars:
        raise ValueError(f"The data variable '{key}' does not exist in the dataset.")

    if key in bounds_vars:
        raise ValueError("Please specify a non-bounds data variable.")

    return dataset[[key] + bounds_vars]


def _get_data_var(dataset: xr.Dataset, key: str) -> xr.DataArray:
    """Get a data variable in the Dataset by key.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    key : str
        The data variable key.

    Returns
    -------
    xr.DataArray
        The data variable.

    Raises
    ------
    KeyError
        If the data variable does not exist in the Dataset.
    """
    dv = dataset.get(key, None)

    if dv is None:
        raise KeyError(f"The data variable '{key}' does not exist in the Dataset.")

    return dv.copy()
