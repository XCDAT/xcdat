"""Module containing geospatial averaging functions."""
from __future__ import annotations

from functools import reduce
from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
    get_args,
)

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from xcdat.axis import (
    _align_lon_bounds_to_360,
    _get_prime_meridian_index,
    get_dim_coords,
    get_dim_keys,
)
from xcdat.dataset import _get_data_var
from xcdat.utils import _if_multidim_dask_array_then_load

#: Type alias for a dictionary of axis keys mapped to their bounds.
AxisWeights = Dict[Hashable, xr.DataArray]
#: Type alias for supported spatial axis keys.
SpatialAxis = Literal["X", "Y"]
SPATIAL_AXES: Tuple[SpatialAxis, ...] = get_args(SpatialAxis)
#: Type alias for a tuple of floats/ints for the regional selection bounds.
RegionAxisBounds = Tuple[float, float]


@xr.register_dataset_accessor("spatial")
class SpatialAccessor:
    """
    An accessor class that provides spatial attributes and methods on xarray
    Datasets through the ``.spatial`` attribute.

    Examples
    --------

    Import SpatialAccessor class:

    >>> import xcdat  # or from xcdat import spatial

    Use SpatialAccessor class:

    >>> ds = xcdat.open_dataset("/path/to/file")
    >>>
    >>> ds.spatial.<attribute>
    >>> ds.spatial.<method>
    >>> ds.spatial.<property>

    Parameters
    ----------
    dataset : xr.Dataset
        A Dataset object.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset

    def average(
        self,
        data_var: str,
        axis: List[SpatialAxis] = ["X", "Y"],
        weights: Union[Literal["generate"], xr.DataArray] = "generate",
        keep_weights: bool = False,
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
    ) -> xr.Dataset:
        """
        Calculates the spatial average for a rectilinear grid over an optionally
        specified regional domain.

        Operations include:

        - If a regional boundary is specified, check to ensure it is within the
          data variable's domain boundary.
        - If axis weights are not provided, get axis weights for standard axis
          domains specified in ``axis``.
        - Adjust weights to conform to the specified regional boundary.
        - Compute spatial weighted average.

        This method requires that the dataset's coordinates have the 'axis'
        attribute set to the keys in ``axis``. For example, the latitude
        coordinates should have its 'axis' attribute set to 'Y' (which is also
        CF-compliant). This 'axis' attribute is used to retrieve the related
        coordinates via `cf_xarray`. Refer to this method's examples for more
        information.

        Parameters
        ----------
        data_var: str
            The name of the data variable inside the dataset to spatially
            average.
        axis : List[SpatialAxis]
            List of axis dimensions to average over, by default ["X", "Y"].
            Valid axis keys include "X" and "Y".
        weights : {"generate", xr.DataArray}, optional
            If "generate", then weights are generated. Otherwise, pass a
            DataArray containing the regional weights used for weighted
            averaging. ``weights`` must include the same spatial axis dimensions
            and have the same dimensional sizes as the data variable, by default
            "generate".
        keep_weights : bool, optional
            If calculating averages using weights, keep the weights in the
            final dataset output, by default False.
        lat_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional latitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound cannot be
            larger than the upper bound, by default None.
        lon_bounds : Optional[RegionAxisBounds], optional
            A tuple of floats/ints for the regional longitude lower and upper
            boundaries. This arg is used when calculating axis weights, but is
            ignored if ``weights`` are supplied. The lower bound can be larger
            than the upper bound (e.g., across the prime meridian, dateline), by
            default None.

        Returns
        -------
        xr.Dataset
            Dataset with the spatially averaged variable.

        Raises
        ------
        KeyError
            If data variable does not exist in the Dataset.

        Examples
        --------

        Check the 'axis' attribute is set on the required coordinates:

        >>> ds.lat.attrs["axis"]
        >>> Y
        >>>
        >>> ds.lon.attrs["axis"]
        >>> X

        Set the 'axis' attribute for the required coordinates if it isn't:

        >>> ds.lat.attrs["axis"] = "Y"
        >>> ds.lon.attrs["axis"] = "X"

        Call spatial averaging method:

        >>> ds.spatial.average(...)

        Get global average time series:

        >>> ts_global = ds.spatial.average("tas", axis=["X", "Y"])["tas"]

        Get time series in Nino 3.4 domain:

        >>> ts_n34 = ds.spatial.average("ts", axis=["X", "Y"],
        >>>     lat_bounds=(-5, 5),
        >>>     lon_bounds=(-170, -120))["ts"]

        Get zonal mean time series:

        >>> ts_zonal = ds.spatial.average("tas", axis=["X"])["tas"]

        Using custom weights for averaging:

        >>> # The shape of the weights must align with the data var.
        >>> self.weights = xr.DataArray(
        >>>     data=np.ones((4, 4)),
        >>>     coords={"lat": self.ds.lat, "lon": self.ds.lon},
        >>>     dims=["lat", "lon"],
        >>> )
        >>>
        >>> ts_global = ds.spatial.average("tas", axis=["X", "Y"],
        >>>     weights=weights)["tas"]
        """
        ds = self._dataset.copy()
        dv = _get_data_var(ds, data_var)
        self._validate_axis_arg(axis)

        if isinstance(weights, str) and weights == "generate":
            if lat_bounds is not None:
                self._validate_region_bounds("Y", lat_bounds)
            if lon_bounds is not None:
                self._validate_region_bounds("X", lon_bounds)
            self._weights = self.get_weights(axis, lat_bounds, lon_bounds, data_var)
        elif isinstance(weights, xr.DataArray):
            self._weights = weights

        self._validate_weights(dv, axis)
        ds[dv.name] = self._averager(dv, axis)

        if keep_weights:
            ds[self._weights.name] = self._weights

        return ds

    def get_weights(
        self,
        axis: List[SpatialAxis],
        lat_bounds: Optional[RegionAxisBounds] = None,
        lon_bounds: Optional[RegionAxisBounds] = None,
        data_var: Optional[str] = None,
    ) -> xr.DataArray:
        """
        Get area weights for specified axis keys and an optional target domain.

        This method first determines the weights for an individual axis based on
        the difference between the upper and lower bound. For latitude the
        weight is determined by the difference of sine(latitude). All axis
        weights are then combined to form a DataArray of weights that can be
        used to perform a weighted (spatial) average.

        If ``lat_bounds`` or ``lon_bounds`` are supplied, then grid cells
        outside this selected regional domain are given zero weight. Grid cells
        that are partially in this domain are given partial weight.

        Parameters
        ----------
        axis : List[SpatialAxis]
            List of axis dimensions to average over.
        lat_bounds : Optional[RegionAxisBounds]
            Tuple of latitude boundaries for regional selection, by default
            None.
        lon_bounds : Optional[RegionAxisBounds]
            Tuple of longitude boundaries for regional selection, by default
            None.
        data_var: Optional[str]
            The key of the data variable, by default None. Pass this argument
            when the dataset has more than one bounds per axis (e.g., "lon"
            and "zlon_bnds" for the "X" axis), or you want weights for a
            specific data variable.

        Returns
        -------
        xr.DataArray
           A DataArray containing the region weights to use during averaging.
           ``weights`` are 1-D and correspond to the specified axes (``axis``)
           in the region.

        Notes
        -----
        This method was developed for rectilinear grids only. ``get_weights()``
        recognizes and operate on latitude and longitude, but could be extended
        to work with other standard geophysical dimensions (e.g., time, depth,
        and pressure).
        """
        Bounds = TypedDict(
            "Bounds", {"weights_method": Callable, "region": Optional[np.ndarray]}
        )

        axis_bounds: Dict[SpatialAxis, Bounds] = {
            "X": {
                "weights_method": self._get_longitude_weights,
                "region": np.array(lon_bounds, dtype="float")
                if lon_bounds is not None
                else None,
            },
            "Y": {
                "weights_method": self._get_latitude_weights,
                "region": np.array(lat_bounds, dtype="float")
                if lat_bounds is not None
                else None,
            },
        }

        axis_weights: AxisWeights = {}
        for key in axis:
            d_bounds = self._dataset.bounds.get_bounds(axis=key, var_key=data_var)

            if isinstance(d_bounds, xr.Dataset):
                raise TypeError(
                    "Generating area weights requires a single bounds per "
                    f"axis, but the dataset has multiple bounds for the '{key}' axis "
                    f"{list(d_bounds.data_vars)}. Pass a `data_var` key "
                    "to reference a specific data variable's axis bounds."
                )

            r_bounds = axis_bounds[key]["region"]

            weights = axis_bounds[key]["weights_method"](d_bounds, r_bounds)
            weights.attrs = d_bounds.attrs
            axis_weights[key] = weights

        weights = self._combine_weights(axis_weights)

        return weights

    def _validate_axis_arg(self, axis: List[SpatialAxis]):
        """
        Validates that the ``axis`` dimension(s) exists in the dataset.

        Parameters
        ----------
        axis : List[SpatialAxis]
            List of axis dimensions to average over.

        Raises
        ------
        ValueError
            If a key in ``axis`` is not a supported value.
        KeyError
            If the dataset does not have coordinates for the ``axis`` dimension,
            or the `axis` attribute is not set for those coordinates.
        """
        for key in axis:
            if key not in SPATIAL_AXES:
                raise ValueError(
                    "Incorrect `axis` argument value. Supported values include: "
                    f"{', '.join(SPATIAL_AXES)}."
                )

            # Check the axis coordinate variable exists in the Dataset.
            get_dim_coords(self._dataset, key)

    def _validate_region_bounds(self, axis: SpatialAxis, bounds: RegionAxisBounds):
        """Validates the ``bounds`` arg based on a set of criteria.

        Parameters
        ----------
        axis : SpatialAxis
            The axis related to the bounds.
        bounds : RegionAxisBounds
            The axis bounds.

        Raises
        ------
        TypeError
            If ``bounds`` is not a tuple.
        ValueError
            If the ``bounds`` has 0 elements or greater than 2 elements.
        TypeError
            If the ``bounds`` lower bound is not a float or integer.
        TypeError
            If the ``bounds`` upper bound is not a float or integer.
        ValueError
            If the ``axis`` is "Y" and the ``bounds`` lower value is larger
            than the upper value.
        """
        if not isinstance(bounds, tuple):
            raise TypeError(
                f"The {axis} regional bounds is not a tuple representing the lower and "
                "upper bounds, (lower, upper)."
            )

        if len(bounds) != 2:
            raise ValueError(
                f"The {axis} regional bounds is not a length of 2 (lower, upper)."
            )

        lower, upper = bounds
        if not isinstance(lower, float) and not isinstance(lower, int):
            raise TypeError(
                f"The regional {axis} lower bound is not a float or an integer."
            )

        if not isinstance(upper, float) and not isinstance(upper, int):
            raise TypeError(
                f"The regional {axis} upper bound is not a float or an integer."
            )

        # For the "Y" axis (latitude), require that the upper bound be larger
        # than the lower bound. Note that this does not apply to the "X" axis
        # (longitude) since it is circular.
        if axis == "Y" and lower >= upper:
            raise ValueError(
                "The regional latitude lower bound is greater than the upper. "
                "Pass a tuple with the format (lower, upper)."
            )

    def _get_longitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: Optional[np.ndarray]
    ) -> xr.DataArray:
        """Gets weights for the longitude axis.

        This method performs longitudinal processing including (in order):

        1. Align the axis orientations of the domain and region bounds to
           (0, 360) to ensure compatibility in the proceeding steps.
        2. Handle grid cells that cross the prime meridian (e.g., [-1, 1]) by
           breaking such grid cells into two (e.g., [0, 1] and [359, 360]) to
           ensure alignment with the (0, 360) axis orientation. This results in
           a bounds axis of length(nlon)+1. The index of the grid cell that
           crosses the prime meridian is returned in order to reduce the length
           of weights to nlon.
        3. Scale the domain down to a region (if selected).
        4. Calculate weights using the domain bounds.
        5. If the prime meridian grid cell exists, use this cell's index to
           handle the weights vector's increased length as a result of the two
           additional grid cells. The extra weights are added to the prime
           meridian grid cell and removed from the weights vector to ensure the
           lengths of the weights and its corresponding domain remain in
           alignment.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the longitude domain.
        region_bounds : Optional[np.ndarray]
            The array of bounds for longitude regional selection.

        Returns
        -------
        xr.DataArray
            The longitude axis weights.

        Raises
        ------
        ValueError
            If the there are multiple instances in which the
            domain_bounds[:, 0] > domain_bounds[:, 1]
        """
        p_meridian_index: Optional[np.ndarray] = None
        d_bounds = domain_bounds.copy()

        pm_cells = np.where(domain_bounds[:, 1] - domain_bounds[:, 0] < 0)[0]
        if len(pm_cells) > 1:
            raise ValueError(
                "More than one longitude bound is out of order. Only one bound "
                "value spanning the prime meridian is permitted in data on "
                "a rectilinear grid."
            )
        d_bounds: xr.DataArray = self._swap_lon_axis(d_bounds, to=360)  # type: ignore
        p_meridian_index = _get_prime_meridian_index(d_bounds)
        if p_meridian_index is not None:
            d_bounds = _align_lon_bounds_to_360(d_bounds, p_meridian_index)

        if region_bounds is not None:
            r_bounds: np.ndarray = self._swap_lon_axis(
                region_bounds, to=360
            )  # type:ignore

            is_region_circular = r_bounds[1] - r_bounds[0] == 0
            if is_region_circular:
                r_bounds = np.array([0.0, 360.0])

            d_bounds = self._scale_domain_to_region(d_bounds, r_bounds)

        weights = self._calculate_weights(d_bounds)
        if p_meridian_index is not None:
            weights[p_meridian_index] = weights[p_meridian_index] + weights[-1]
            weights = weights[0:-1]

        return weights

    def _get_latitude_weights(
        self, domain_bounds: xr.DataArray, region_bounds: Optional[np.ndarray]
    ) -> xr.DataArray:
        """Gets weights for the latitude axis.

        This method scales the domain to a region (if selected). It also scales
        the area between two lines of latitude as the difference of the sine of
        latitude bounds.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for the latitude domain.
        region_bounds : Optional[np.ndarray]
            The array of bounds for latitude regional selection.

        Returns
        -------
        xr.DataArray
            The latitude axis weights.
        """
        if region_bounds is not None:
            domain_bounds = self._scale_domain_to_region(domain_bounds, region_bounds)

        d_bounds = np.sin(np.radians(domain_bounds))
        weights = self._calculate_weights(d_bounds)
        return weights

    def _calculate_weights(self, domain_bounds: xr.DataArray):
        """Calculate weights for the domain.

        This method takes the absolute difference between the upper and lower
        bound values to calculate weights.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The array of bounds for a domain.

        Returns
        -------
        xr.DataArray
            The weights for an axes.
        """
        return np.abs(domain_bounds[:, 1] - domain_bounds[:, 0])

    def _swap_lon_axis(
        self, lon: Union[xr.DataArray, np.ndarray], to: Literal[180, 360]
    ) -> Union[xr.DataArray, np.ndarray]:
        """Swap the longitude axis orientation.

        Parameters
        ----------
        lon : Union[xr.DataArray, np.ndarray]
             Longitude values to convert.
        to : Literal[180, 360]
            Axis orientation to convert to, either 180 [-180, 180) or 360
            [0, 360).

        Returns
        -------
        Union[xr.DataArray, np.ndarray]
            Converted longitude values.

        Notes
        -----
        This does not reorder the values in any way; it only converts the values
        in-place between longitude conventions [-180, 180) or [0, 360).
        """
        lon_swap = lon.copy()

        if isinstance(lon_swap, xr.DataArray):
            _if_multidim_dask_array_then_load(lon_swap)

        # Must set keep_attrs=True or the xarray DataArray attrs will get
        # dropped. This has no affect on NumPy arrays.
        with xr.set_options(keep_attrs=True):
            if to == 180:
                lon_swap = ((lon_swap + 180) % 360) - 180
            elif to == 360:
                lon_swap = lon_swap % 360
            else:
                raise ValueError(
                    "Only longitude axis orientation 180 or 360 is supported."
                )

        return lon_swap

    def _scale_domain_to_region(
        self, domain_bounds: xr.DataArray, region_bounds: np.ndarray
    ) -> xr.DataArray:
        """
        Scale domain bounds to conform to a regional selection in order to
        calculate spatial weights.

        Axis weights are determined by the difference between the upper
        and lower boundary. If a region is selected, the grid cell
        bounds outside the selected region are adjusted using this method
        so that the grid cell bounds match the selected region bounds. The
        effect of this adjustment is to give partial weight to grid cells
        that are partially in the selected regional domain and zero weight
        to grid cells outside the selected domain.

        Parameters
        ----------
        domain_bounds : xr.DataArray
            The domain's bounds.
        region_bounds : np.ndarray
            The region bounds that the domain bounds are scaled down to.

        Returns
        -------
        xr.DataArray
            Scaled dimension bounds based on regional selection.

        Notes
        -----
        If a lower regional selection bound exceeds the upper selection bound,
        this algorithm assumes that the axis is longitude and the user is
        specifying a region that includes the prime meridian. The lower
        selection bound should not exceed the upper bound for latitude.
        """
        d_bounds = domain_bounds.copy()
        r_bounds = region_bounds.copy()

        _if_multidim_dask_array_then_load(d_bounds)

        # Since longitude is circular, the logic depends on whether the region
        # spans across the prime meridian or not. If a region does not include
        # the prime meridian, then grid cells between the upper/lower region
        # domain values are given weight. If the prime meridian is included in
        # the domain (e.g., for a left bound of 300 degrees and a right bound
        # of 20, then the grid cells in between the region bounds (20 and 300)
        # are given zero weight (or partial weight if the grid bounds overlap
        # with the region bounds).
        if r_bounds[1] >= r_bounds[0]:
            # Case 1 (simple case): not wrapping around prime meridian (or
            # latitude axis).
            # Adjustments for above / right of region.
            d_bounds[d_bounds[:, 0] > r_bounds[1], 0] = r_bounds[1]
            d_bounds[d_bounds[:, 1] > r_bounds[1], 1] = r_bounds[1]
            # Adjustments for below / left of region.
            d_bounds[d_bounds[:, 0] < r_bounds[0], 0] = r_bounds[0]
            d_bounds[d_bounds[:, 1] < r_bounds[0], 1] = r_bounds[0]

        else:
            # Case 2: wrapping around prime meridian [for longitude only]
            domain_lowers = d_bounds[:, 0]
            domain_uppers = d_bounds[:, 1]
            region_lower, region_upper = r_bounds

            # Grid cell straddling lower boundary.
            inds = np.where(
                (domain_lowers < region_lower) & (domain_uppers > region_lower)
            )[0]
            d_bounds[inds, 0] = region_lower

            # Grid cells in between boundaries (i.e., outside selection domain).
            inds = np.where(
                (domain_lowers >= region_upper) & (domain_uppers <= region_lower)
            )[0]
            # Set upper and lower grid cell boundaries to upper edge of
            # regional domain. This will mean the grid cell upper and lower
            # boundary are equal. Therefore their difference will be zero
            # and their weight will also be zero.
            d_bounds[inds, :] = region_upper

            # Grid cell straddling upper boundary.
            inds = np.where(
                (domain_lowers < region_upper) & (domain_uppers > region_upper)
            )[0]
            d_bounds[inds, 1] = r_bounds[1]

        return d_bounds

    def _combine_weights(self, axis_weights: AxisWeights) -> xr.DataArray:
        """Generically rescales axis weights for a given region.

        This method creates an n-dimensional weighting array by performing
        matrix multiplication for a list of specified axis keys using a
        dictionary of axis weights.

        Parameters
        ----------
        axis_weights : AxisWeights
            Dictionary of axis weights, where key is axis and value is the
            corresponding DataArray of weights.

        Returns
        -------
        xr.DataArray
            A DataArray containing the region weights to use during averaging.
            ``weights`` are 1-D and correspond to the specified axis keys
            (``axis``) in the region.
        """
        region_weights = reduce((lambda x, y: x * y), axis_weights.values())

        coord_keys = sorted(region_weights.dims)  # type: ignore
        region_weights.name = "_".join(coord_keys) + "_wts"  # type: ignore

        return region_weights

    def _validate_weights(self, data_var: xr.DataArray, axis: List[SpatialAxis]):
        """Validates the ``weights`` arg based on a set of criteria.

        This methods checks for the dimensional alignment between the
        ``weights`` and ``data_var``. It assumes that ``data_var`` has the same
        keys that are specified  in ``axis``, which has already been validated
        using ``self._validate_axis()`` in ``self.average()``.

        Parameters
        ----------
        data_var : xr.DataArray
            The data variable used for validation with user supplied weights.
        axis : List[SpatialAxis]
            List of axes dimension(s) average over.
        weights : xr.DataArray
            A DataArray containing the region area weights for averaging.
            ``weights`` must include the same spatial axis dimensions found in
            ``axis`` and ``data_var``, and the same axis dims sizes as
            ``data_var``.
        Raises
        ------
        KeyError
            If ``weights`` does not include the latitude dimension.
        KeyError
            If ``weights`` does not include the longitude dimension.
        ValueError
            If the axis dimension sizes between ``weights`` and ``data_var``
            are misaligned.
        """

        # Check the weights includes the same axis as the data variable.
        for key in axis:
            dim_name = get_dim_keys(data_var, key)
            if dim_name not in self._weights.dims:
                raise KeyError(
                    f"The weights DataArray does not include an {key} axis, or the "
                    "dimension names are not the same."
                )

        # Check the weight dim sizes equal data var dim sizes.
        dim_sizes = {key: data_var.sizes[key] for key in self._weights.sizes.keys()}
        for dim, size in self._weights.sizes.items():
            if size != dim_sizes[dim]:
                raise ValueError(
                    f"The axis dimension sizes between supplied `weights` {dict(self._weights.sizes)} "
                    f"and the data variable {dim_sizes} are misaligned."
                )

    def _averager(self, data_var: xr.DataArray, axis: List[SpatialAxis]):
        """Perform a weighted average of a data variable.

        This method assumes all specified keys in ``axis`` exists in the data
        variable. Validation for this criteria is performed in
        ``_validate_weights()``.

        Operations include:

        - Masked (missing) data receives zero weight.
        - Perform weighted average over user-specified axes/axis.

        Parameters
        ----------
        data_var : xr.DataArray
            Data variable inside a Dataset.
        axis : List[SpatialAxis]
            List of axis dimensions to average over.

        Returns
        -------
        xr.DataArray
            Variable that has been reduced via a weighted average.

        Notes
        -----
        ``weights`` must be a DataArray and cannot contain missing values.
        Missing values are replaced with 0 using ``weights.fillna(0)``.
        """
        weights = self._weights.fillna(0)

        dim = []
        for key in axis:
            dim.append(get_dim_keys(data_var, key))

        with xr.set_options(keep_attrs=True):
            weighted_mean = data_var.cf.weighted(weights).mean(dim=dim)

        return weighted_mean
