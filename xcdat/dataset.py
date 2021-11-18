"""Dataset module for functions related to an xarray.Dataset."""
from glob import glob
from typing import Any, Dict, Hashable, List, Optional, Union

import pandas as pd
import xarray as xr
from typing_extensions import Literal

from xcdat import bounds  # noqa: F401
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)


def open_dataset(
    path: str, data_var: Optional[str] = None, **kwargs: Dict[str, Any]
) -> xr.Dataset:
    """Wrapper for ``xarray.open_dataset()`` that applies common operations.

    Operations include:

    - Decode both CF and non-CF compliant time units if the Dataset has a time
      dimension
    - Add missing bounds for supported axis
    - Option to limit the Dataset to a single regular (non-bounds) data
      variable, while retaining any bounds data variables

    ``decode_times`` is statically set to ``False``. This enables a check
    for whether the units in the time dimension (if it exists) contains CF or
    non-CF compliant units, which determines if manual decoding is necessary.

    Parameters
    ----------
    path : str
        Path to Dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    kwargs : Dict[str, Any]
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

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path")

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path", data_var="tas")

    Keep multiple variables in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path", data_var=["ts", "tas"])
    """
    ds = xr.open_dataset(path, decode_times=False, **kwargs)
    ds = infer_or_keep_var(ds, data_var)

    if ds.cf.dims.get("T") is not None:
        ds = decode_time_units(ds)

    ds = ds.bounds.add_missing_bounds()
    return ds


def open_mfdataset(
    paths: Union[str, List[str]],
    data_var: Optional[str] = None,
    data_vars: Union[Literal["minimal", "different", "all"], List[str]] = "minimal",
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wrapper for ``xarray.open_mfdataset()`` that applies common operations.

    Operations include:

    - Decode both CF and non-CF compliant time units if the Dataset has a time
      dimension
    - Fill missing bounds for supported axis
    - Option to limit the Dataset to a single regular (non-bounds) data
      variable, while retaining any bounds data variables

    ``data_vars`` defaults to ``"minimal"``, which concatenates data variables
    in a manner where only data variables in which the dimension already appears
    are included. For example, the time dimension will not be concatenated to
    the dimensions of non-time data variables such as "lat_bnds" or "lon_bnds".
    `"minimal"` is required for some XCDAT functions, including spatial
    averaging where a reduction is performed using the lat/lon bounds.

    ``decode_times`` is statically set to ``False``. This enables a check
    for whether the units in the time dimension (if it exists) contains CF or
    non-CF compliant units, which determines if manual decoding is necessary.

    Parameters
    ----------
    path : Union[str, List[str]]
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an
        explicit list of files to open. Paths can be given as strings or as
        pathlib Paths. If concatenation along more than one dimension is desired,
        then ``paths`` must be a nested list-of-lists (see ``combine_nested``
        for details). (A string glob will be expanded to a 1-dimensional list.)
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset, by default None.
    data_vars: Union[Literal["minimal", "different", "all"], List[str]], optional
        These data variables will be concatenated together:
          * "minimal": Only data variables in which the dimension already
            appears are included, default.
          * "different": Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * "all": All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the "minimal" data variables.
    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset``. Refer to
        the [2]_ xarray docs for accepted keyword arguments.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_mfdataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [2] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_mfdataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"])

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"], data_var="tas")

    Keep multiple variables in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"], data_var=["ts", "tas"])
    """
    # check if time axis is cf_compliant
    cf_compliant = _check_dataset_for_cf_compliant_time(paths)

    # if cf_compliant, let xarray decode the time units
    # otherwise, decode using decode_time_units
    if cf_compliant:
        ds = xr.open_mfdataset(paths, decode_times=True, data_vars=data_vars, **kwargs)
    else:
        ds = xr.open_mfdataset(paths, decode_times=False, data_vars=data_vars, **kwargs)
        if ds.cf.dims.get("T") is not None:
            ds = decode_time_units(ds)

    ds = infer_or_keep_var(ds, data_var)

    ds = ds.bounds.add_missing_bounds()
    return ds


def infer_or_keep_var(dataset: xr.Dataset, data_var: Optional[str]) -> xr.Dataset:
    """Infer the data variable(s) or keep a specific one in the Dataset.

    If ``data_var`` is None, then this function checks the number of
    regular (non-bounds) data variables in the Dataset. If there is a single
    regular data var, then it will add an 'xcdat_infer' attr pointing to it in
    the Dataset. XCDAT APIs can then call `get_inferred_var()` to get the data
    var linked to the 'xcdat_infer' attr. If there are multiple regular data
    variables, the 'xcdat_infer' attr is not set and the Dataset is returned
    as is.

    If ``data_var`` is not None, then this function checks if the ``data_var``
    exists in the Dataset and if it is a regular data var. If those checks pass,
    it will subset the Dataset to retain that ``data_var`` and all bounds data
    vars. An 'xcdat_infer' attr pointing to the ``data_var`` is also added
    to the Dataset.

    This utility function is useful for designing XCDAT APIs with an optional
    ``data_var`` kwarg. If ``data_var`` is None, an inference to the desired
    data var is performed with a call to this function. Otherwise, perform the
    API operation explicitly on ``data_var``.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    data_var: Optional[str], optional
        The key of the data variable to keep in the Dataset.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    KeyError
        If the specified data variable is not found in the Dataset.
    KeyError
        If the user specifies a bounds variable to keep.
    """
    ds = dataset.copy()
    # Make sure the "xcdat_infer" attr is None because a Dataset may be written
    # with this attr already set.
    ds.attrs["xcdat_infer"] = None

    all_vars = ds.data_vars.keys()
    bounds_vars = ds.bounds.names
    regular_vars: List[Hashable] = list(set(all_vars) ^ set(bounds_vars))

    if len(regular_vars) == 0:
        logger.debug("This dataset only contains bounds data variables.")

    if data_var is None:
        if len(regular_vars) == 1:
            ds.attrs["xcdat_infer"] = regular_vars[0]
        elif len(regular_vars) > 1:
            regular_vars_str = ", ".join(
                f"'{var}'" for var in sorted(regular_vars)  # type:ignore
            )
            logger.debug(
                "This dataset contains more than one regular data variable "
                f"({regular_vars_str}). If desired, pass the `data_var` kwarg to "
                "reduce down to one regular data var."
            )
    if data_var is not None:
        if data_var not in all_vars:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )
        if data_var in bounds_vars:
            raise KeyError("Please specify a regular (non-bounds) data variable.")

        ds = dataset[[data_var] + bounds_vars]
        ds.attrs["xcdat_infer"] = data_var

    return ds


def decode_time_units(dataset: xr.Dataset):
    """Decodes both CF and non-CF compliant time units.

    ``xarray`` uses the ``cftime`` module, which only supports CF compliant
    time units [4]_. As a result, opening datasets with non-CF compliant
    time units (months and years) will throw an error if ``decode_times=True``.

    This function works around this issue by first checking if the time units
    are CF or non-CF compliant. Datasets with CF compliant time units are passed
    to ``xarray.decode_cf``. Datasets with non-CF compliant time units are
    manually decoded by extracting the units and reference date, which are used
    to generate an array of datetime values.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with non-decoded CF/non-CF compliant time units.

    Returns
    -------
    xr.Dataset
        Dataset with decoded time units.

    Notes
    -----
    .. [4] https://unidata.github.io/cftime/api.html#cftime.num2date

    Examples
    --------

    Decode non-CF compliant time units in a Dataset:

    >>> from xcdat.dataset import decode_time_units
    >>> ds = xr.open_dataset("file_path", decode_times=False)
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
    >>> ds = decode_time_units(ds)
    >>> ds.time
    <xarray.DataArray 'time' (time: 3)>
    array(['2000-01-01T00:00:00.000000000', '2001-01-01T00:00:00.000000000',
        '2002-01-01T00:00:00.000000000'], dtype='datetime64[ns]')
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-01 2001-01-01 2002-01-01
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time

    View time coordinate encoding information:

    >>> ds.time.encoding
    {'source': None, 'dtype': dtype('int64'), 'original_shape': (3,), 'units':
    'years since 2000-01-01', 'calendar': 'proleptic_gregorian'}
    """
    time = dataset["time"]
    units_attr = time.attrs.get("units")

    if units_attr is None:
        raise KeyError(
            "No 'units' attribute found for time coordinate. Make sure to open "
            "the dataset with `decode_times=False`."
        )

    units, reference_date = units_attr.split(" since ")
    non_cf_units_to_freq = {"months": "MS", "years": "YS"}

    cf_compliant = units not in non_cf_units_to_freq.keys()
    if cf_compliant:
        dataset = xr.decode_cf(dataset, decode_times=True)
    else:
        # NOTE: The "calendar" attribute for units consisting of "months" or
        # "years" is not factored when generating date ranges. The number of
        # days in a month is not factored.
        decoded_time = xr.DataArray(
            data=pd.date_range(
                start=reference_date,
                periods=time.size,
                freq=non_cf_units_to_freq[units],
            ),
            dims=["time"],
            attrs=dataset["time"].attrs,
        )
        decoded_time.encoding = {
            "source": dataset.encoding.get("source"),
            "dtype": time.dtype,
            "original_shape": decoded_time.shape,
            "units": units_attr,
            # pandas.date_range() returns "proleptic_gregorian" by default
            "calendar": "proleptic_gregorian",
        }

        dataset = dataset.assign_coords({"time": decoded_time})
    return dataset


def _check_dataset_for_cf_compliant_time(path: Union[str, List[str]]):
    """Determine if a dataset has cf_compliant time

    Operations include:

    - Open the file / dataset (in the case of multi-file datasets, only open
      one file)
    - Determine the time units and whether they are cf-compliant
    - Return a Boolean (None if the time axis or time units do not exist)

    Parameters
    ----------
    path : Union[str, List[str]]
        Either a file (``"file.nc"``), a string glob in the form
        ``"path/to/my/files/*.nc"``, or an explicit list of files to open.
        Paths can be given as strings or as pathlib Paths. If concatenation
        along more than one dimension is desired, then ``paths`` must be a
        nested list-of-lists (see ``combine_nested`` for details). (A string
        glob will be expanded to a 1-dimensional list.)

    Returns
    -------
    Boolean
        True if dataset is cf_compliant or False if not
        Returns None if time or time units are not present

    Notes
    -----
    This function only checks one file of multifile datasets (for performance).

    """
    # non-cf compliant units handled by xcdat
    # Note: Should this be defined more globally? Is it possible to do the
    # opposite (e.g., get the list of cf_compliant units and check that)?
    non_cf_units_to_freq = ["months", "years"]

    # Get one example file to check
    # Note: This doesn't handle pathlib paths or a list of lists
    if type(path) == str:
        if "*" in path:
            fn1 = glob(path)[0]
        else:
            fn1 = path
    else:
        fn1 = path[0]

    # Open one file
    ds = xr.open_dataset(fn1, decode_times=False)
    # if there is no time dimension return None for the time units
    # else get the time units
    if ds.cf.dims.get("T") is None:
        cf_compliant = None
    else:
        time = ds["time"]
        units_attr = time.attrs.get("units")
        units, reference_date = units_attr.split(" since ")
        cf_compliant = units not in non_cf_units_to_freq
    ds.close()

    return cf_compliant


def get_inferred_var(dataset: xr.Dataset) -> xr.DataArray:
    """Gets the inferred data variable that is tagged in the Dataset.

    This function looks for the "xcdat_infer" attribute pointing
    to the desired data var in the Dataset, which can be set through
    ``xcdat.open_dataset()``, ``xcdat.open_mf_dataset()``, or manually.

    This utility function is useful for designing XCDAT APIs with an optional
    ``data_var`` kwarg. If ``data_var`` is None, an inference to the desired
    data var is performed with a call to this function. Otherwise, perform the
    API operation explicitly on ``data_var``.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.

    Returns
    -------
    xr.DataArray
        The inferred data variable.

    Raises
    ------
    KeyError
        If the 'xcdat_infer' attr is not set in the Dataset.
    KeyError
        If the 'xcdat_infer' attr points to a non-existent data var.
    KeyError
        If the 'xcdat_infer' attr points to a bounds data var.
    """
    inferred_var = dataset.attrs.get("xcdat_infer", None)
    bounds_vars = dataset.bounds.names

    if inferred_var is None:
        raise KeyError(
            "Dataset attr 'xcdat_infer' is not set so the desired data variable "
            "cannot be inferred. You must pass the `data_var` kwarg to this operation."
        )
    else:
        data_var = dataset.get(inferred_var, None)
        if data_var is None:
            raise KeyError(
                "Dataset attr 'xcdat_infer' is set to non-existent data variable, "
                f"'{inferred_var}'. Either pass the `data_var` kwarg to this operation, "
                "or set 'xcdat_infer' to a regular (non-bounds) data variable."
            )
        if inferred_var in bounds_vars:
            raise KeyError(
                "Dataset attr `xcdat_infer` is set to the bounds data variable, "
                f"'{inferred_var}'. Either pass the `data_var` kwarg, or set "
                "'xcdat_infer' to a regular (non-bounds) data variable."
            )

        logger.debug(
            f"The data variable '{data_var.name}' was inferred from the Dataset attr "
            "'xcdat_infer' for this operation."
        )
        return data_var.copy()
