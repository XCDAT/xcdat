import importlib
import json
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr
from dask.array.core import Array


def compare_datasets(ds1: xr.Dataset, ds2: xr.Dataset) -> Dict[str, List[str]]:
    """Compares the keys and values of two datasets.

    This utility function is especially useful for debugging tests that
    involve comparing two Dataset objects for being identical or equal.

    Checks include:

    - Unique keys - keys that exist only in one of the two datasets.
    - Non-identical - keys whose values have the same dimension, coordinates,
      values, name, attributes, and attributes on all coordinates.
    - Non-equal keys - keys whose values have the same dimension, coordinates,
      and values, but not necessarily the same attributes. Key values that are
      non-equal will also be non-identical.

    Parameters
    ----------
    ds1 : xr.Dataset
        The first Dataset.
    ds2 : xr.Dataset
        The second Dataset.

    Returns
    -------
    Dict[str, Union[List[str]]]
        A dictionary mapping unique, non-identical, and
        non-equal keys in both Datasets.
    """
    results = {
        "unique_coords": list(ds1.coords.keys() ^ ds2.coords.keys()),
        "unique_data_vars": list(ds1.data_vars.keys() ^ ds2.data_vars.keys()),
        "nonidentical_coords": [],
        "nonidentical_data_vars": [],
        "nonequal_coords": [],
        "nonequal_data_vars": [],
    }

    ds_keys = {
        "coords": ds1.coords.keys() & ds2.coords.keys(),
        "data_vars": ds1.data_vars.keys() & ds2.data_vars.keys(),
    }
    for key_type, keys in ds_keys.items():
        for key in keys:
            identical = ds1[key].identical(ds2[key])
            equals = ds1[key].equals(ds2[key])

            if not identical:
                results[f"nonidentical_{key_type}"].append(key)
            if not equals:
                results[f"nonequal_{key_type}"].append(key)

    return results


def str_to_bool(attr: str) -> bool:
    """Converts bool string to bool.

    netCDF files can only store attributes with a type of str, Number, ndarray,
    number, list, or tuple.

    xCDAT methods store boolean attributes as strings. This function will
    convert such attributes back to booleans.

    Parameters
    ----------
    attr : str
        The boolean attribute as type str.

    Returns
    -------
    bool
        The boolean attribute as type bool.
    """
    if attr != "True" and attr != "False":
        raise ValueError(
            "The attribute is not a string representation of a Python"
            "bool ('True' or 'False')"
        )

    bool_attr = json.loads(attr.lower())
    return bool_attr


def _has_module(modname: str) -> bool:  # pragma: no cover
    """Checks if the specified module is installed in the Python environment.

    Parameters
    ----------
    modname : str
        The name of the module.

    Returns
    -------
    bool
    """
    try:
        importlib.import_module(modname)
        has = True
    except ImportError:
        has = False

    return has


def _if_multidim_dask_array_then_load(
    obj: Union[xr.DataArray, xr.Dataset]
) -> Optional[Union[xr.DataArray, xr.Dataset]]:
    """
    If the underlying array for an xr.DataArray or xr.Dataset is a
    multidimensional, lazy Dask Array, load it into an in-memory NumPy array.

    This function must be called before manipulating values in a
    multidimensional Dask Array, which xarray does not support directly.
    Otherwise, it raises `NotImplementedError xarray can't set arrays with
    multiple array indices to dask yet`.

    Parameters
    ----------
    obj : Union[xr.DataArray, xr.Dataset]
        The xr.DataArray or xr.Dataset. If the xarray object is chunked,
        the underlying array will be a Dask Array.
    """
    if isinstance(obj.data, Array) and obj.ndim > 1:
        return obj.load()

    return None


def mask_var_with_weight_threshold(
    dv: xr.DataArray, weights: xr.DataArray, min_weight: float
) -> xr.DataArray:
    """Mask values that do not meet the minimum weight threshold using np.nan.

    This function is useful for cases where the weighting of data might be
    skewed based on the availability of data. For example, if one season in a
    time series has more significantly more missing data than other seasons, it
    can result in inaccurate calculations of climatologies. Masking values that
    do not meet the minimum weight threshold ensures more accurate calculations.

    Parameters
    ----------
    dv : xr.DataArray
        The weighted variable.
    weights : xr.DataArray
        A DataArray containing either the regional or temporal weights used for
        weighted averaging. ``weights`` must include the same axis dimensions
        and dimensional sizes as the data variable.
    min_weight : float
        Fraction of data coverage (i..e, weight) needed to return a
        spatial average value. Value must range from 0 to 1.

    Returns
    -------
    xr.DataArray
        The variable with the minimum weight threshold applied.
    """
    masked_weights = _get_masked_weights(dv, weights)

    # Sum all weights, including zero for missing values.
    dim = weights.dims
    weight_sum_all = weights.sum(dim=dim)
    weight_sum_masked = masked_weights.sum(dim=dim)

    # Get fraction of the available weight.
    frac = weight_sum_masked / weight_sum_all

    # Nan out values that don't meet specified weight threshold.
    dv_new = xr.where(frac >= min_weight, dv, np.nan, keep_attrs=True)
    dv_new.name = dv.name

    return dv_new


def _get_masked_weights(dv: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """Get weights with missing data (`np.nan`) receiving no weight (zero).

    Parameters
    ----------
    dv : xr.DataArray
        The variable.
    weights : xr.DataArray
        A DataArray containing either the regional or temporal weights used for
        weighted averaging. ``weights`` must include the same axis dimensions
        and dimensional sizes as the data variable.

    Returns
    -------
    xr.DataArray
        The masked weights.
    """
    masked_weights = xr.where(dv.copy().isnull(), 0.0, weights)

    return masked_weights


def _validate_min_weight(min_weight: float | None) -> float:
    """Validate the ``min_weight`` value.

    Parameters
    ----------
    min_weight : float | None
        Fraction of data coverage (i..e, weight) needed to return a
        spatial average value. Value must range from 0 to 1.

    Returns
    -------
    float
        The required weight percentage.

    Raises
    ------
    ValueError
        If the `min_weight` argument is less than 0.
    ValueError
        If the `min_weight` argument is greater than 1.
    """
    if min_weight is None:
        return 0.0
    elif min_weight < 0.0:
        raise ValueError(
            "min_weight argument is less than 0. " "min_weight must be between 0 and 1."
        )
    elif min_weight > 1.0:
        raise ValueError(
            "min_weight argument is greater than 1. "
            "min_weight must be between 0 and 1."
        )

    return min_weight
