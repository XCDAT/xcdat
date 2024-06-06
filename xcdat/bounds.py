"""Bounds module for functions related to coordinate bounds."""

from __future__ import annotations

import collections
import datetime
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import cf_xarray as cfxr  # noqa: F401
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.cftime_offsets import get_date_type
from xarray.core.common import contains_cftime_datetimes

from xcdat._logger import _setup_custom_logger
from xcdat.axis import CF_ATTR_MAP, CFAxisKey, get_dim_coords
from xcdat.dataset import _get_data_var
from xcdat.temporal import (
    _contains_datetime_like_objects,
    _get_datetime_like_type,
    _infer_freq,
)

logger = _setup_custom_logger(__name__)


@xr.register_dataset_accessor("bounds")
class BoundsAccessor:
    """
    An accessor class that provides bounds attributes and methods on xarray
    Datasets through the ``.bounds`` attribute.

    Examples
    --------

    Import BoundsAccessor class:

    >>> import xcdat  # or from xcdat import bounds

    Use BoundsAccessor class:

    >>> ds = xcdat.open_dataset("/path/to/file")
    >>>
    >>> ds.bounds.<attribute>
    >>> ds.bounds.<method>
    >>> ds.bounds.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.

    Examples
    ---------
    Import:

    >>> from xcdat import bounds

    Return dictionary of axis and coordinate keys mapped to bounds:

    >>> ds.bounds.map

    Return list of keys for bounds data variables:

    >>> ds.bounds.keys

    Add missing coordinate bounds for supported axes in the Dataset:

    >>> ds = ds.bounds.add_missing_bounds(axes=["X", "Y", "T"])

    Get coordinate bounds if they exist:

    >>> lat_bounds = ds.bounds.get_bounds("Y")
    >>> lon_bounds = ds.bounds.get_bounds("X")
    >>> time_bounds = ds.bounds.get_bounds("T")

    Add coordinate bounds for a specific axis if they don't exist:

    >>> ds = ds.bounds.add_bounds("Y")
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    @property
    def map(self) -> Dict[str, Optional[xr.DataArray]]:
        """Returns a map of axis and coordinates keys to their bounds.

        The dictionary provides all valid CF compliant keys for axis and
        coordinates. For example, latitude will includes keys for "lat",
        "latitude", and "Y".

        Returns
        -------
        Dict[str, Optional[xr.DataArray]]
            Dictionary mapping axis and coordinate keys to their bounds.
        """
        ds = self._dataset

        bounds: Dict[str, Optional[xr.DataArray]] = {}
        for axis, bounds_keys in ds.cf.bounds.items():
            bound = ds.get(bounds_keys[0], None)
            bounds[axis] = bound

        return collections.OrderedDict(sorted(bounds.items()))

    @property
    def keys(self) -> List[str]:
        """Returns a list of keys for the bounds data variables in the Dataset.

        Returns
        -------
        List[str]
            A list of sorted bounds data variable keys.
        """
        return sorted(
            list(
                {
                    key
                    for bound_keys in self._dataset.cf.bounds.values()
                    for key in bound_keys
                }
            )
        )

    def add_missing_bounds(  # noqa: C901
        self, axes: List[CFAxisKey] | Tuple[CFAxisKey, ...] = ("X", "Y", "T")
    ) -> xr.Dataset:
        """Adds missing coordinate bounds for supported axes in the Dataset.

        This function loops through the Dataset's axes and attempts to adds
        bounds to its coordinates if they don't exist. "X", "Y" , and "Z" axes
        bounds are the midpoints between coordinates. "T" axis bounds are based
        on the time frequency of the coordinates.

        An axis must meet the following criteria to add bounds for it, otherwise
        they are ignored:

        1. Axis is either X", "Y", "T", or "Z"
        2. Coordinates are a single dimension, not multidimensional
        3. Coordinates are a length > 1 (not singleton)
        4. Bounds must not already exist

           * Coordinates are mapped to bounds using the "bounds" attr. For
             example, bounds exist if ``ds.time.attrs["bounds"]`` is set to
             ``"time_bnds"`` and ``ds.time_bnds`` is present in the dataset.

        5. For the "T" axis, its coordinates must be composed of datetime-like
           objects (``np.datetime64`` or ``cftime``). This method designed to
           operate on time axes that have constant temporal resolution with
           annual, monthly, daily, or sub-daily time frequencies. Alternate
           frequencies (e.g., pentad) are not supported.

        Parameters
        ----------
        axes : List[CFAxesKey] | Tuple[CFAxisKey, ...]
            List of CF axes that function should operate on, by default
            ("X", "Y", "T"). Options include "X", "Y", "T", or "Z".

        Returns
        -------
        xr.Dataset
        """
        ds = self._dataset.copy()

        for axis in axes:
            try:
                coords = get_dim_coords(ds, axis)
            except KeyError:
                continue

            # In xarray, ancillary singleton coordinates that aren't related to
            # axis can still be attached to dimension coordinates (e.g.,
            # "height" is attached to "time"). We ignore these singleton
            # coordinates to avoid adding bounds for them.
            coords = self._drop_ancillary_singleton_coords(coords)
            for coord in coords.coords.values():
                try:
                    self.get_bounds(axis, str(coord.name))

                    continue
                except KeyError:
                    pass

                try:
                    if axis in ["X", "Y", "Z"]:
                        bounds = create_bounds(axis, coord)
                    elif axis == "T":
                        bounds = self._create_time_bounds(coord)
                except (ValueError, TypeError) as e:
                    logger.warning(e)
                    continue

                ds[bounds.name] = bounds
                ds[coord.name].attrs["bounds"] = bounds.name

        return ds

    def get_bounds(
        self, axis: CFAxisKey, var_key: Optional[str] = None
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Gets coordinate bounds.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", "Z").
        var_key: Optional[str]
            The key of the coordinate or data variable to get axis bounds for.
            This parameter is useful if you only want the single bounds
            DataArray related to the axis on the variable (e.g., "tas" has
            a "lat" dimension and you want "lat_bnds").

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            A Dataset of N bounds variables, or a single bounds variable
            DataArray.

        Raises
        ------
        ValueError
            If an incorrect ``axis`` argument is passed.

        KeyError:
            If bounds were not found for the specific ``axis``.
        """
        self._validate_axis_arg(axis)

        if var_key is None:
            # Get all bounds keys in the Dataset for this axis.
            bounds_keys = self._get_bounds_keys(axis)
        else:
            # Get the obj in the Dataset using the key.
            obj = _get_data_var(self._dataset, key=var_key)
            bounds_keys = self._get_bounds_from_attr(obj, axis)

        if len(bounds_keys) == 0:
            raise KeyError(
                f"No bounds data variables were found for the '{axis}' axis. Make sure "
                "the dataset has bound data vars and their names match the 'bounds' "
                "attributes found on their related time coordinate variables. "
                "Alternatively, you can add bounds with `ds.bounds.add_missing_bounds()` "
                "or `ds.bounds.add_bounds()`."
            )

        bounds: Union[xr.Dataset, xr.DataArray] = self._dataset[
            bounds_keys if len(bounds_keys) > 1 else bounds_keys[0]
        ].copy()

        return bounds

    def add_bounds(self, axis: CFAxisKey) -> xr.Dataset:
        """Add bounds for an axis using its coordinates as midpoints.

        This method loops over the axis's coordinate variables and attempts to
        add bounds for each of them if they don't exist. Each coordinate point
        is the midpoint between their lower and upper bounds.

        To add bounds for an axis its coordinates must meet the following
        criteria, otherwise an error is thrown:

        1. Axis is either X", "Y", "T", or "Z"
        2. Coordinates are single dimensional, not multidimensional
        3. Coordinates are a length > 1 (not singleton)
        4. Bounds must not already exist

           * Coordinates are mapped to bounds using the "bounds" attr. For
             example, bounds exist if ``ds.time.attrs["bounds"]`` is set to
             ``"time_bnds"`` and ``ds.time_bnds`` is present in the dataset.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", "Z").

        Returns
        -------
        xr.Dataset
            The dataset with bounds added.

        Raises
        """
        ds = self._dataset.copy()
        self._validate_axis_arg(axis)

        coord_vars: Union[xr.DataArray, xr.Dataset] = get_dim_coords(
            self._dataset, axis
        )
        # In xarray, ancillary singleton coordinates that aren't related to the
        # axis can still be attached to dimension coordinates. For example,
        # if the "height" singleton exists, it will be attached to "time".
        # Singleton coordinates are dropped in order to add bounds for the axis.
        coord_vars = self._drop_ancillary_singleton_coords(coord_vars)

        for coord in coord_vars.coords.values():
            # Check if the coord var has a "bounds" attr and the bounds actually
            # exist in the Dataset. If it does not, then add the bounds.
            try:
                bounds_key = ds[coord.name].attrs["bounds"]
                ds[bounds_key]

                continue
            except KeyError:
                bounds = create_bounds(axis, coord)

                ds[bounds.name] = bounds
                ds[coord.name].attrs["bounds"] = bounds.name

        return ds

    def add_time_bounds(
        self,
        method: Literal["freq", "midpoint"],
        freq: Optional[Literal["year", "month", "day", "hour"]] = None,
        daily_subfreq: Optional[Literal[1, 2, 3, 4, 6, 8, 12, 24]] = None,
        end_of_month: bool = False,
    ) -> xr.Dataset:
        """Add bounds for an axis using its coordinate points.

        This method designed to operate on time axes that have constant temporal
        resolution with annual, monthly, daily, or sub-daily time frequencies.
        Alternate frequencies (e.g., pentad) are not supported. It loops over
        the time axis coordinate variables and attempts to add bounds for each
        of them if they don't exist.

        To add time bounds for the time axis, its coordinates must be the
        following criteria:

        1. Coordinates are single dimensional, not multidimensional
        2. Coordinates are a length > 1 (not singleton)
        3. Bounds must not already exist

           * Coordinates are mapped to bounds using the "bounds" attr. For
             example, bounds exist if ``ds.time.attrs["bounds"]`` is set to
             ``"time_bnds"`` and ``ds.time_bnds`` is present in the dataset.

        4. If ``method=freq``, coordinates must be composed of datetime-like
           objects (``np.datetime64`` or ``cftime``)

        Parameters
        ----------
        method : {"freq", "midpoint"}
            The method for creating time bounds for time coordinates, either
            "freq" or "midpoint".

            * "freq": Create time bounds as the start and end of each timestep's
              period using either the inferred or specified time frequency
              (``freq`` parameter). For example, the time bounds will be the
              start and end of each month for each monthly coordinate point.
            * "midpoint": Create time bounds using time coordinates as the
              midpoint between their upper and lower bounds.

        freq : {"year", "month", "day", "hour"}, optional
            If ``method="freq"``, this parameter specifies the time frequency
            for creating time bounds. By default None, which infers the
            frequency using the time coordinates.
        daily_subfreq : {1, 2, 3, 4, 6, 8, 12, 24}, optional
            If ``freq=="hour"``, this parameter sets the number of timepoints
            per day for time bounds, by default None.

            * ``daily_subfreq=None`` infers the daily time frequency from the
              time coordinates.
            * ``daily_subfreq=1`` is daily
            * ``daily_subfreq=2`` is twice daily
            * ``daily_subfreq=4`` is 6-hourly
            * ``daily_subfreq=8`` is 3-hourly
            * ``daily_subfreq=12`` is 2-hourly
            * ``daily_subfreq=24`` is hourly

        end_of_month : bool, optional
            If ``freq=="month"``, this flag notes that the timepoint is saved
            at the end of the monthly interval (see Note), by default False.

            * Some timepoints are saved at the end of the interval, e.g., Feb. 1
              00:00 for the time interval Jan. 1 00:00 - Feb. 1 00:00. Since this
              method determines the month and year from the time vector, the
              bounds will be set incorrectly if the timepoint is set to the end of
              the time interval. For these cases, set ``end_of_month=True``.

        Returns
        -------
        xr.Dataset
            The dataset with time bounds added.
        """
        ds = self._dataset.copy()
        coord_vars: Union[xr.DataArray, xr.Dataset] = get_dim_coords(self._dataset, "T")
        # In xarray, ancillary singleton coordinates that aren't related to axis
        # can still be attached to dimension coordinates (e.g., "height" is
        # attached to "time"). We ignore these singleton coordinates to avoid
        # adding bounds for them.
        coord_vars = self._drop_ancillary_singleton_coords(coord_vars)

        for coord in coord_vars.coords.values():
            # Check if the coord var has a "bounds" attr and the bounds actually
            # exist in the Dataset. If it does not, then add the bounds.
            try:
                bounds_key = ds[coord.name].attrs["bounds"]
                ds[bounds_key]

                continue
            except KeyError:
                if method == "freq":
                    bounds = self._create_time_bounds(
                        coord, freq, daily_subfreq, end_of_month
                    )
                elif method == "midpoint":
                    bounds = create_bounds("T", coord)

                ds[bounds.name] = bounds
                ds[coord.name].attrs["bounds"] = bounds.name

        return ds

    def _drop_ancillary_singleton_coords(
        self, coord_vars: Union[xr.Dataset, xr.DataArray]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """Drop ancillary singleton coordinates from dimension coordinates.

        Xarray coordinate variables retain all coordinates from the parent
        object. This means if singleton coordinates exist, they are attached to
        dimension coordinates as ancillary coordinates. For example, the
        "height" singleton coordinate will be attached to "time" coordinates
        even though "height" is related to the "Z" axis, not the "T" axis.
        Refer to [1]_ for more info on this Xarray behavior.

        This is an undesirable behavior in xCDAT because the add bounds methods
        loop over coordinates related to an axis and attempts to add bounds if
        they don't exist. If ancillary coordinates are present, "ValueError:
        Cannot generate bounds for coordinate variable 'height' which has a
        length <= 1 (singleton)" is raised. For the purpose of adding bounds, we
        temporarily drop any ancillary singletons from dimension coordinates
        before looping over those coordinates. Ancillary singletons will still
        be present in the final Dataset object to maintain the Dataset's
        integrity.

        Parameters
        ----------
        coord_vars : Union[xr.Dataset, xr.DataArray]
            The dimension coordinate variables with ancillary coordinates (if
            they exist).

        Returns
        -------
        Union[xr.Dataset, xr.DataArray]
            The dimension coordinate variables with ancillary coordinates
            dropped (if they exist).

        References
        ----------
        .. [1] https://github.com/pydata/xarray/issues/6196
        """
        dims = coord_vars.dims
        coords = coord_vars.coords.keys()

        singleton_coords = set(dims) ^ set(coords)

        if len(singleton_coords) > 0:
            return coord_vars.drop_vars(singleton_coords)

        return coord_vars

    def _get_bounds_keys(self, axis: CFAxisKey) -> List[str]:
        """Get bounds keys for an axis's coordinate variables in the dataset.

        This function attempts to map bounds to an axis using ``cf_xarray``
        and its interpretation of the CF "bounds" attribute.

        Parameters
        ----------
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", or "Z").

        Returns
        -------
        List[str]
            The axis bounds key(s).
        """
        cf_method = self._dataset.cf.bounds
        cf_attrs = CF_ATTR_MAP[axis]

        keys: List[str] = []

        try:
            keys = keys + cf_method[cf_attrs["axis"]]
        except KeyError:
            pass

        try:
            keys = cf_method[cf_attrs["coordinate"]]
        except KeyError:
            pass

        keys_from_attr = self._get_bounds_from_attr(self._dataset, axis)
        keys = keys + keys_from_attr

        return list(set(keys))

    def _get_bounds_from_attr(
        self, obj: xr.DataArray | xr.Dataset, axis: CFAxisKey
    ) -> List[str]:
        """Retrieve bounds attribute keys from the given xarray object.

        This method extracts the "bounds" attribute keys from the coordinates
        of the specified axis in the provided xarray DataArray or Dataset.

        Parameters:
        -----------
        obj : xr.DataArray | xr.Dataset
            The xarray object from which to retrieve the bounds attribute keys.
        axis : CFAxisKey
            The CF axis key ("X", "Y", "T", or "Z").

        Returns:
        --------
        List[str]
            A list of bounds attribute keys found in the coordinates of the
            specified axis. Otherwise, an empty list is returned.
        """
        coords_obj = get_dim_coords(obj, axis)
        bounds_keys: List[str] = []

        if isinstance(coords_obj, xr.DataArray):
            bounds_keys = self._extract_bounds_key(coords_obj, bounds_keys)
        elif isinstance(coords_obj, xr.Dataset):
            for coord in coords_obj.coords.values():
                bounds_keys = self._extract_bounds_key(coord, bounds_keys)

        return bounds_keys

    def _extract_bounds_key(
        self, coords_obj: xr.DataArray, bounds_keys: List[str]
    ) -> List[str]:
        bnds_key = coords_obj.attrs.get("bounds")

        if bnds_key is not None:
            bounds_keys.append(bnds_key)

        return bounds_keys

    def _create_time_bounds(  # noqa: C901
        self,
        time: xr.DataArray,
        freq: Optional[Literal["year", "month", "day", "hour"]] = None,
        daily_subfreq: Optional[Literal[1, 2, 3, 4, 6, 8, 12, 24]] = None,
        end_of_month: bool = False,
    ) -> xr.DataArray:
        """Creates time bounds for each timestep of the time coordinate axis.

        This method creates time bounds as the start and end of each timestep's
        period using either the inferred or specified time frequency (``freq``
        parameter). For example, the time bounds will be the start and end of
        each month for each monthly coordinate point.

        Parameters
        ----------
        time : xr.DataArray
            The temporal coordinate variable for the axis.
        freq : {"year", "month", "day", "hour"}, optional
            The time frequency for creating time bounds, by default None (infer
            the frequency).
        daily_subfreq : {1, 2, 3, 4, 6, 8, 12, 24}, optional
            If ``freq=="hour"``, this parameter sets the number of timepoints
            per day for bounds, by default None. If greater than 1, sub-daily
            bounds are created.

            * ``daily_subfreq=None`` infers the freq from the time coords (default)
            * ``daily_subfreq=1`` is daily
            * ``daily_subfreq=2`` is twice daily
            * ``daily_subfreq=4`` is 6-hourly
            * ``daily_subfreq=8`` is 3-hourly
            * ``daily_subfreq=12`` is 2-hourly
            * ``daily_subfreq=24`` is hourly

        end_of_month : bool, optional
            If `freq=="month"``, this flag notes that the timepoint is saved
            at the end of the monthly interval (see Note), by default False.

        Returns
        -------
        xr.DataArray
            A DataArray storing bounds for the time axis.

        Raises
        ------
        ValueError
            If coordinates are a singleton.
        TypeError
            If time coordinates are not composed of datetime-like objects.

        Note
        ----
        Some timepoints are saved at the end of the interval, e.g., Feb. 1 00:00
        for the time interval Jan. 1 00:00 - Feb. 1 00:00. Since this function
        determines the month and year from the time vector, the bounds will be set
        incorrectly if the timepoint is set to the end of the time interval. For
        these cases, set ``end_of_month=True``.
        """
        is_singleton = time.size <= 1
        if is_singleton:
            raise ValueError(
                f"Cannot generate bounds for coordinate variable '{time.name}'"
                " which has a length <= 1 (singleton)."
            )

        if not _contains_datetime_like_objects(time):
            raise TypeError(
                f"Bounds cannot be created for '{time.name}' coordinates because it is "
                "not decoded as `cftime.datetime` or `np.datetime`. Try decoding "
                f"'{time.name}' first then adding bounds."
            )

        freq = _infer_freq(time) if freq is None else freq  # type: ignore
        timesteps = time.values

        # Determine the object type for creating time bounds based on the
        # object type/dtype of the time coordinates.
        if _get_datetime_like_type(time) == np.datetime64:
            # Cast time values from `np.datetime64` to `pd.Timestamp` (a
            # sub-class of `np.datetime64`) in order to get access to the
            # pandas time/date components which simplifies creating bounds.
            # https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
            timesteps = pd.to_datetime(timesteps)
            obj_type = pd.Timestamp
        elif _get_datetime_like_type(time) == cftime.datetime:
            calendar = time.encoding["calendar"]
            obj_type = get_date_type(calendar)

        if freq == "year":
            time_bnds = self._create_yearly_time_bounds(timesteps, obj_type)
        elif freq == "month":
            time_bnds = self._create_monthly_time_bounds(
                timesteps, obj_type, end_of_month
            )
        elif freq == "day":
            time_bnds = self._create_daily_time_bounds(timesteps, obj_type)
        elif freq == "hour":
            # Determine the daily frequency for generating time bounds.
            if daily_subfreq is None:
                diff = time.values[1] - time.values[0]
                hrs = diff.seconds / 3600
                daily_subfreq = int(24 / hrs)  # type: ignore

            time_bnds = self._create_daily_time_bounds(
                timesteps,
                obj_type,
                freq=daily_subfreq,  # type: ignore
            )

        # Create the bounds data array
        da_time_bnds = xr.DataArray(
            name=f"{time.name}_bnds",
            data=time_bnds,
            coords={time.name: time},
            dims=[*time.dims, "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        return da_time_bnds

    def _create_yearly_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the year.

        Bounds for each timestep correspond to Jan. 1 00:00:00 of the year of the
        timestep and Jan. 1 00:00:00 of the subsequent year.

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.

        Returns
        -------
        List[Union[cftime.datetime, pd.Timestamp]]
            A list of time bound values.
        """
        time_bnds: List[cftime.datetime] = []

        for step in timesteps:
            year = step.year

            l_bnd = obj_type(year, 1, 1, 0, 0)
            u_bnd = obj_type(year + 1, 1, 1, 0, 0)

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _create_monthly_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
        end_of_month: bool = False,
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the month.

        Bounds for each timestep correspond to 00:00:00 on the first of the month
        and 00:00:00 on the first of the subsequent month.

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.
        end_of_month : bool, optional
            Flag to note that the timepoint is saved at the end of the monthly
            interval (see Note), by default False.

        Returns
        -------
        List[Union[cftime.datetime, pd.Timestamp]]
            A list of time bound values.

        Note
        ----
        Some timepoints are saved at the end of the interval, e.g., Feb. 1 00:00
        for the time interval Jan. 1 00:00 - Feb. 1 00:00. Since this function
        determines the month and year from the time vector, the bounds will be set
        incorrectly if the timepoint is set to the end of the time interval. For
        these cases, set ``end_of_month=True``.
        """
        time_bnds = []

        for step in timesteps:
            # If end of time interval and first day of year then subtract one
            # month so bounds will be calculated correctly.
            if (end_of_month) & (step.day < 2):
                step = self._add_months_to_timestep(step, obj_type, delta=-1)

            year, month = step.year, step.month

            l_bnd = obj_type(year, month, 1, 0, 0)
            u_bnd = self._add_months_to_timestep(l_bnd, obj_type, delta=1)

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _add_months_to_timestep(
        self,
        timestep: Union[cftime.datetime, pd.Timestamp],
        obj_type: Union[cftime.datetime, pd.Timestamp],
        delta: int,
    ) -> Union[cftime.datetime, pd.Timestamp]:
        """Adds delta month(s) to a timestep.

        The delta value can be positive or negative (for subtraction). Refer to
        [4]_ for logic.

        Parameters
        ----------
        timestep : Union[cftime.datime, pd.Timestamp]
            A timestep represented as ``cftime.datetime`` or ``pd.Timestamp``.
        obj_type : Union[cftime.datetime, pd.Timestamp]
                The object type for time bounds based on the dtype of
                ``timestep``.
        delta : int
            Integer months to be added to times (can be positive or negative)

        Returns
        -------
        Union[cftime.datetime, pd.Timestamp]

        References
        ----------
        .. [4] https://stackoverflow.com/a/4131114
        """
        # Compute the new month and year with the delta month(s).
        month = timestep.month - 1 + delta
        year = timestep.year + month // 12
        month = month % 12 + 1

        # Re-use existing hour/minute/second/microsecond.
        hour = timestep.hour
        minute = timestep.minute
        second = timestep.second
        microsecond = timestep.microsecond

        # If day is greater than days in month use days in month as day.
        day = timestep.day
        dim = obj_type(year, month, 1).daysinmonth
        day = min(day, dim)

        # Create the new timestep.
        new_timestep = obj_type(year, month, day, hour, minute, second, microsecond)

        return new_timestep

    def _create_daily_time_bounds(
        self,
        timesteps: np.ndarray,
        obj_type: Union[cftime.datetime, pd.Timestamp],
        freq: Literal[1, 2, 3, 4, 6, 8, 12, 24] = 1,
    ) -> List[Union[cftime.datetime, pd.Timestamp]]:
        """Creates time bounds for each timestep with the start and end of the day.

        Bounds for each timestep corresponds to 00:00:00 timepoint on the
        current day and 00:00:00 on the subsequent day.

        If time steps are sub-daily, then the bounds will begin at 00:00 and end
        at 00:00 of the following day. For example, for 3-hourly data, the
        bounds would be::

            [
                ["01/01/2000 00:00", "01/01/2000 03:00"],
                ["01/01/2000 03:00", "01/01/2000 06:00"],
                ...
                ["01/01/2000 21:00", "02/01/2000 00:00"],
            ]

        Parameters
        ----------
        timesteps : np.ndarray
            An array of timesteps, represented as either `cftime.datetime` or
            `pd.Timestamp` (casted from `np.datetime64[ns]` to support pandas
            time/date components).
        obj_type : Union[cftime.datetime, pd.Timestamp]
            The object type for time bounds based on the dtype of
            ``time_values``.
        freq : {1, 2, 3, 4, 6, 8, 12, 24}, optional
            Number of timepoints per day, by default 1. If greater than 1, sub-daily
            bounds are created.

            * ``freq=1`` is daily (default)
            * ``freq=2`` is twice daily
            * ``freq=4`` is 6-hourly
            * ``freq=8`` is 3-hourly
            * ``freq=12`` is 2-hourly
            * ``freq=24`` is hourly

        Returns
        -------
        List[Union[cftime.datetime, pd.Timestamp]]
            A list of time bound values.

        Raises
        ------
        ValueError
            If an incorrect ``freq`` argument is passed. Should be 1, 2, 3, 4, 6, 8,
            12, or 24.

        Notes
        -----
        This function is intended to reproduce CDAT's ``setAxisTimeBoundsDaily``
        method [5]_.

        References
        ----------
        .. [5] https://github.com/CDAT/cdutil/blob/master/cdutil/times.py#L1093
        """
        if (freq > 24) | (np.mod(24, freq)):
            raise ValueError(
                "Incorrect `freq` argument. Supported values include 1, 2, 3, 4, 6, 8, 12, "
                "and 24."
            )

        time_bnds = []

        for step in timesteps:
            y, m, d, h = step.year, step.month, step.day, step.hour

            for f in range(freq):
                if f * (24 // freq) <= h < (f + 1) * (24 // freq):
                    l_bnd = obj_type(y, m, d, f * (24 // freq))
                    u_bnd = l_bnd + datetime.timedelta(hours=(24 // freq))

            time_bnds.append([l_bnd, u_bnd])

        return time_bnds

    def _validate_axis_arg(self, axis: CFAxisKey):
        cf_axis_keys = CF_ATTR_MAP.keys()

        if axis not in cf_axis_keys:
            keys = ", ".join(f"'{key}'" for key in cf_axis_keys)
            raise ValueError(
                f"Incorrect 'axis' argument value. Supported values include {keys}."
            )

        get_dim_coords(self._dataset, axis)


def create_bounds(axis: CFAxisKey, coord_var: xr.DataArray) -> xr.DataArray:
    """Creates bounds for an axis using coordinate points as midpoints.

    This method uses each coordinate point as the midpoint between its
    lower and upper bound.

    Parameters
    ----------
    axis: CFAxisKey
        The CF axis key ("X", "Y", "T" ,"Z").
    coord_var : xr.DataArray
        The coordinate variable for the axis.

    Returns
    -------
    xr.DataArray
        The axis coordinate bounds.

    Raises
    ------
    ValueError
        If coords dimensions does not equal 1.

    Notes
    -----
    Based on [2]_ ``iris.coords._guess_bounds`` and [3]_
    ``cf_xarray.accessor.add_bounds``.

    For temporal coordinates ``create_bounds`` will attempt to set the
    bounds to the start and end of each time step's period. Time axes are
    expected to be composed of ``cftime`` objects.

    References
    ----------
    .. [2] https://scitools-iris.readthedocs.io/en/stable/generated/api/iris/coords.html#iris.coords.AuxCoord.guess_bounds

    .. [3] https://cf-xarray.readthedocs.io/en/latest/generated/xarray.Dataset.cf.add_bounds.html#
    """
    is_singleton = coord_var.size <= 1
    if is_singleton:
        raise ValueError(
            f"Cannot generate bounds for coordinate variable '{coord_var.name}'"
            " which has a length <= 1 (singleton)."
        )

    # Retrieve coordinate dimension to calculate the diffs between points.
    dim = coord_var.dims[0]
    diffs = coord_var.diff(dim).values

    # Add beginning and end points to account for lower and upper bounds.
    # np.array of string values with `dtype="timedelta64[ns]"`
    diffs = np.insert(diffs, 0, diffs[0])
    diffs = np.append(diffs, diffs[-1])

    # `cftime` objects only support arithmetic using `timedelta` objects, so
    # the values of  `diffs` must be casted from `dtype="timedelta64[ns]"`
    # to `timedelta` objects.
    if axis == "T" and contains_cftime_datetimes(xr.as_variable(coord_var)):
        diffs = pd.to_timedelta(diffs)

    # FIXME: These lines produces the warning: `PerformanceWarning:
    # Adding/subtracting object-dtype array to TimedeltaArray not
    # vectorized` after converting diffs to `timedelta`. I (Tom) was not
    # able to find an alternative, vectorized solution at the time of this
    # implementation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

        width = 0.5
        lower_bounds = coord_var - diffs[:-1] * width
        upper_bounds = coord_var + diffs[1:] * (1 - width)

    # Transpose both bound arrays into a 2D array.
    data = np.array([lower_bounds, upper_bounds]).transpose()

    # Clip latitude bounds at (-90, 90)
    if axis == "Y":
        units = coord_var.attrs.get("units")

        if units is None:
            coord_var.attrs["units"] = "degrees_north"
            logger.warning(
                f"The '{coord_var.name}' coordinate variable is missing "
                "a 'units' attribute. Assuming 'units' is 'degrees_north'."
            )
        elif "degree" not in units.lower():
            raise ValueError(
                f"The {coord_var.name} coord variable has a 'units' attribute that "
                "is not in degrees."
            )

        if (coord_var >= -90).all() and (coord_var <= 90).all():
            np.clip(data, -90, 90, out=data)

    # Create the bounds data variable and add it to the Dataset.
    bounds = xr.DataArray(
        name=f"{coord_var.name}_bnds",
        data=data,
        coords={coord_var.name: coord_var},
        dims=[*coord_var.dims, "bnds"],
        attrs={"xcdat_bounds": "True"},
    )

    return bounds
