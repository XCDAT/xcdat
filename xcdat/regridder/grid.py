from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from xcdat.axis import COORD_DEFAULT_ATTRS, VAR_NAME_MAP, CFAxisKey, get_dim_coords
from xcdat.bounds import create_bounds

# First 50 zeros for the bessel function
# Taken from https://github.com/CDAT/cdms/blob/dd41a8dd3b5bac10a4bfdf6e56f6465e11efc51d/regrid2/Src/_regridmodule.c#L3390-L3402
BESSEL_LOOKUP = [
    2.4048255577,
    5.5200781103,
    8.6537279129,
    11.7915344391,
    14.9309177086,
    18.0710639679,
    21.2116366299,
    24.3524715308,
    27.493479132,
    30.6346064684,
    33.7758202136,
    36.9170983537,
    40.0584257646,
    43.1997917132,
    46.3411883717,
    49.4826098974,
    52.6240518411,
    55.765510755,
    58.9069839261,
    62.0484691902,
    65.1899648002,
    68.3314693299,
    71.4729816036,
    74.6145006437,
    77.7560256304,
    80.8975558711,
    84.0390907769,
    87.1806298436,
    90.3221726372,
    93.4637187819,
    96.605267951,
    99.7468198587,
    102.8883742542,
    106.0299309165,
    109.1714896498,
    112.3130502805,
    115.4546126537,
    118.5961766309,
    121.737742088,
    124.8793089132,
    128.0208770059,
    131.1624462752,
    134.3040166383,
    137.4455880203,
    140.5871603528,
    143.7287335737,
    146.8703076258,
    150.011882457,
    153.1534580192,
    156.2950342685,
]

# Error used for legendre polinomial
# Taken from CDAT's regrid2 module https://github.com/CDAT/cdms/blob/3f8c7baa359f428628a666652ecf361764dc7b7a/regrid2/Src/_regridmodule.c#L3229
EPS = 1e-14
# Constant used in first estimate for legendre polinomial
# Taken from CDAT's regrid2 module https://github.com/CDAT/cdms/blob/3f8c7baa359f428628a666652ecf361764dc7b7a/regrid2/Src/_regridmodule.c#L3254-L3256
ESTIMATE_CONST = 0.25 * (1.0 - np.power(2.0 / np.pi, 2))


def create_gaussian_grid(nlats: int) -> xr.Dataset:
    """
    Creates a grid with Gaussian latitudes and uniform longitudes.

    Parameters
    ----------
    nlats : int
        Number of latitudes.

    Returns
    -------
    xr.Dataset
        Dataset with new grid, containing Gaussian latitudes.

    Examples
    --------
    Create grid with 32 latitudes:

    >>> xcdat.regridder.grid.create_gaussian_grid(32)
    """
    lat_bnds, lat = _create_gaussian_axis(nlats)

    lon = _create_uniform_axis(0.0, 360.0, (360.0 / (2.0 * nlats)))

    return create_grid(x=create_axis("lon", lon), y=(lat, lat_bnds))


def _create_gaussian_axis(nlats: int) -> Tuple[xr.DataArray, xr.DataArray]:
    """Create Gaussian axis.

    Creates a Gaussian axis of `nlats`.

    Parameters
    ----------
    nlats : int
        Number of latitude points.

    Returns
    -------
    xr.DataArray
        Containing the latitude bounds.

    xr.DataArray
        Containing the latitude axis.
    """
    mid = int(np.floor(nlats / 2))

    points, weights = _gaussian_axis(mid, nlats)

    bounds = np.zeros((nlats + 1))
    bounds[0], bounds[-1] = 1.0, -1.0

    for i in range(1, mid + 1):
        bounds[i] = bounds[i - 1] - weights[i - 1]
        bounds[nlats - i] = -bounds[i]

    points_data = (180.0 / np.pi) * np.arcsin(points)

    points_da = xr.DataArray(
        name="lat",
        data=points_data,
        dims=["lat"],
        attrs={
            "units": "degrees_north",
            "axis": "Y",
            "bounds": "lat_bnds",
        },
    )

    bounds = (180.0 / np.pi) * np.arcsin(bounds)  # type: ignore

    bounds_data = np.zeros((points.shape[0], 2))
    bounds_data[:, 0] = bounds[:-1]
    bounds_data[:, 1] = bounds[1:]

    bounds_da = xr.DataArray(
        name="lat_bnds",
        data=bounds_data,
        dims=["lat", "bnds"],
        coords={
            "lat": points_da,
        },
    )

    return bounds_da, points_da


def _gaussian_axis(mid: int, nlats: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the bounds and weights for a Guassian axis.


    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3310

    Parameters
    ----------
    mid : int
        mid
    nlats : int
        Number of latitude points to calculate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        First `np.ndarray` contains the angles of the bounds and the second contains the weights.
    """
    points = _bessel_function_zeros(mid + 1)

    points = np.pad(points, (0, nlats - len(points)))

    weights = np.zeros_like(points)

    for x in range(int(nlats / 2 + 1)):
        zero_poly, first_poly, second_poly = _legendre_polinomial(points[x], nlats)

        points[x] = zero_poly

        weights[x] = (
            (1.0 - zero_poly * zero_poly)
            * 2.0
            / ((nlats * first_poly) * (nlats * first_poly))
        )

    points[mid if nlats % 2 == 0 else mid + 1 :] = -1.0 * np.flip(points[:mid])

    weights[mid if nlats % 2 == 0 else mid + 1 :] = np.flip(weights[:mid])

    if nlats % 2 != 0:
        weights[mid + 1] = 0.0

        mid_weight = 2.0 / np.power(nlats, 2)

        for x in range(2, nlats + 1, 2):
            mid_weight = (mid_weight * np.power(nlats, 2)) / np.power(x - 1, 2)

            weights[mid + 1] = mid_weight

    return points, weights


def _bessel_function_zeros(n: int) -> np.ndarray:
    """Zeros of Bessel function.

    Calculates `n` zeros of the Bessel function.

    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://en.wikipedia.org/wiki/Bessel_function
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3387
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3251

    Parameters
    ----------
    n : int
        Number of zeros to calculate.

    Returns
    -------
    np.ndarray
        Array containing `n` zeros of the Bessel function.
    """
    values = np.zeros(n)

    lookup_n = min(n, 50)

    values[:lookup_n] = BESSEL_LOOKUP[:lookup_n]

    # interpolate remaining values
    if n > 50:
        for x in range(50, n):
            values[x] = values[x - 1] + np.pi

    return values


def _legendre_polinomial(bessel_zero: int, nlats: int) -> Tuple[float, float, float]:
    """Legendre_polynomials.

    Calculates the third legendre polynomial.

    Math is based on CDAT implementation in regrid2 module.

    Related documents:
        - https://en.wikipedia.org/wiki/Legendre_polynomials
        - https://github.com/CDAT/cdms/blob/cda99c21098ad01d88c27c6010d4affc8f621863/regrid2/Src/_regridmodule.c#L3291

    Parameters
    ----------
    bessel_zero : int
        Bessel zero used to calculate the third legendre polynomial.
    nlats : int
        Number of lats.

    Returns
    -------
    Tuple[float, float, float]
        First, second and third legendre polynomial.
    """
    zero_poly = np.cos(bessel_zero / np.sqrt(np.power(nlats + 0.5, 2) + ESTIMATE_CONST))

    for _ in range(10):
        second_poly = 1.0
        first_poly = zero_poly

        for y in range(2, nlats + 1):
            new_poly = (
                (2.0 * y - 1.0) * zero_poly * first_poly - (y - 1.0) * second_poly
            ) / y
            second_poly = first_poly
            first_poly = new_poly

        first_poly = second_poly
        poly_change = (nlats * (first_poly - zero_poly * new_poly)) / (
            1.0 - zero_poly * zero_poly
        )
        poly_slope = new_poly / poly_change
        zero_poly -= poly_slope
        abs_poly_change = np.fabs(poly_slope)

        if abs_poly_change <= EPS:
            break

    return zero_poly, first_poly, second_poly


def create_uniform_grid(
    lat_start: float,
    lat_stop: float,
    lat_delta: float,
    lon_start: float,
    lon_stop: float,
    lon_delta: float,
) -> xr.Dataset:
    """
    Creates a uniform rectilinear grid and sets appropriate
    the attributes for the lat/lon axis.

    Parameters
    ----------
    lat_start : float
        First latitude.
    lat_stop : float
        Last latitude.
    lat_delta : float
        Difference between two points of axis.
    lon_start : float
        First longitude.
    lon_stop : float
        Last longitude.
    lon_delta : float
        Difference between two points of axis.

    Returns
    -------
    xr.Dataset
        Dataset with uniform lat/lon grid.

    Examples
    --------
    Create 4x5 uniform grid:

    >>> xcdat.regridder.grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)
    """
    lat = _create_uniform_axis(lat_start, lat_stop + 0.0001, lat_delta)

    lon = _create_uniform_axis(lon_start, lon_stop + 0.0001, lon_delta)

    return create_grid(x=create_axis("lon", lon), y=create_axis("lat", lat))


def _create_uniform_axis(start: float, stop: float, delta: float) -> np.ndarray:
    """Create a uniform axis.

    Start and stop are inclusive.

    Parameters
    ----------
    start : float
        Start value of the array.
    stop : float
        Stop value of the array.
    delta : float
        Change between each value in the array.

    Returns
    -------
    np.ndarray
        Array containing axis values.
    """
    return np.arange(start, stop + 0.0001, delta)


def create_global_mean_grid(grid: xr.Dataset) -> xr.Dataset:
    """Creates a global mean grid.

    Bounds are expected to be present in `grid`.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing the global mean grid.
    """
    lat = get_dim_coords(grid, "Y")
    _validate_grid_has_single_axis_dim("X", lat)

    lat_data = np.array([(lat[0] + lat[-1]) / 2.0])
    lat_bnds = grid.bounds.get_bounds("Y", var_key=lat.name)
    lat_bnds = np.array([[lat_bnds[0, 0], lat_bnds[-1, 1]]])

    lon = get_dim_coords(grid, "X")
    _validate_grid_has_single_axis_dim("Y", lon)

    lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    lon_bnds = grid.bounds.get_bounds("X", var_key=lon.name)
    lon_bnds = np.array([[lon_bnds[0, 0], lon_bnds[-1, 1]]])

    return create_grid(
        x=create_axis("lon", lon_data, bounds=lon_bnds),
        y=create_axis("lat", lat_data, lat_bnds),
    )


def create_zonal_grid(grid: xr.Dataset) -> xr.Dataset:
    """Creates a zonal grid.

    Bounds are expected to be present in `grid`.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing a zonal grid.
    """
    lon = get_dim_coords(grid, "X")
    _validate_grid_has_single_axis_dim("X", lon)

    out_lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    lon_bnds = grid.bounds.get_bounds("X", var_key=lon.name)
    lon_bnds = np.array([[lon_bnds[0, 0], lon_bnds[-1, 1]]])

    lat = get_dim_coords(grid, "Y")
    _validate_grid_has_single_axis_dim("Y", lat)

    lat_bnds = grid.bounds.get_bounds("Y", var_key=lat.name)

    # Ignore `Argument 1 to "create_grid" has incompatible type
    # "Union[Dataset, DataArray]"; expected "Union[ndarray[Any, Any], DataArray]"
    # mypy(error)` because this arg is validated to be a DataArray beforehand.
    return create_grid(
        x=create_axis("lon", out_lon_data, bounds=lon_bnds),
        y=create_axis("lat", lat, bounds=lat_bnds),  # type: ignore
    )


def create_grid(
    x: xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None = None,
    y: xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None = None,
    z: xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None = None,
    attrs: Optional[Dict[str, str]] = None,
) -> xr.Dataset:
    """Creates a grid dataset using the specified axes.

    Parameters
    ----------
    x : xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None
        An optional dataarray or tuple of a datarray with optional bounds to use
        for the "X" axis, by default None.
    y : xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None = None,
        An optional dataarray or tuple of a datarray with optional bounds to use
        for the "Y" axis, by default None.
    z : xr.DataArray | Tuple[xr.DataArray, xr.DataArray | None] | None
        An optional dataarray or tuple of a datarray with optional bounds to use
        for the "Z" axis, by default None.
    attrs : Optional[Dict[str, str]]
        Custom attributes to be added to the generated `xr.Dataset`.

    Returns
    -------
    xr.Dataset
        Dataset with grid axes.

    Examples
    --------
    Create uniform 2.5 x 2.5 degree grid using ``create_axis``:

    >>> # NOTE: `create_axis` returns (axis, bnds)
    >>> lat_axis = create_axis("lat", np.arange(-90, 90, 2.5))
    >>> lon_axis = create_axis("lon", np.arange(1.25, 360, 2.5))
    >>>
    >>> grid = create_grid(x=lon_axis, y=lat_axis)

    With custom attributes:

    >>> grid = create_grid(
    >>>    x=lon_axis, y=lat_axis, attrs={"created": str(datetime.date.today())}
    >>> )

    Create grid using existing `xr.DataArray`'s:

    >>> lat = xr.DataArray(...)
    >>> lon = xr.DataArray(...)
    >>>
    >>> grid = create_grid(x=lon, x=lat)

    With existing bounds:

    >>> lat_bnds = xr.DataArray(...)
    >>> lon_bnds = xr.DataArray(...)
    >>>
    >>> grid = create_grid(x=(lat, lat_bnds), y=(lon, lon_bnds))

    Create vertical grid:

    >>> z = create_axis(
    >>>   "lev", np.linspace(1000, 1, 20), attrs={"units": "meters", "positive": "down"}
    >>> )
    >>> grid = create_grid(z=z)
    """
    if np.all([item is None for item in (x, y, z)]):
        raise ValueError("Must pass at least 1 axis to create a grid.")

    axes = {"x": x, "y": y, "z": z}
    ds = xr.Dataset(attrs={} if attrs is None else attrs.copy())

    for axis, item in axes.items():
        if item is None:
            continue

        if isinstance(item, (tuple, list)):
            if len(item) != 2:
                raise ValueError(
                    f"Argument {axis!r} should be an xr.DataArray representing "
                    "coordinates or a tuple (xr.DataArray, xr.DataArray) representing "
                    "coordinates and bounds."
                )

            coords = item[0].copy(deep=True)

            if item[1] is not None:
                bnds = item[1].copy(deep=True)

                coords.attrs["bounds"] = bnds.name

                ds = ds.assign({bnds.name: bnds})
        else:
            coords = item.copy(deep=True)

        ds = ds.assign_coords({coords.name: coords})

    return ds


def create_axis(
    name: str,
    data: Union[List[Union[int, float]], np.ndarray],
    bounds: Optional[Union[List[List[Union[int, float]]], np.ndarray]] = None,
    generate_bounds: Optional[bool] = True,
    attrs: Optional[Dict[str, str]] = None,
) -> Tuple[xr.DataArray, Optional[xr.DataArray]]:
    """Creates an axis and optional bounds.


    Parameters
    ----------
    name : str
        The CF standard name for the axis (e.g., "longitude", "latitude",
        "height"). xCDAT also accepts additional names such as "lon", "lat",
        and "lev". Refer to ``xcdat.axis.VAR_NAME_MAP`` for accepted names.
    data : Union[List[Union[int, float]], np.ndarray]
        1-D axis data consisting of integers or floats.
    bounds : Optional[Union[List[List[Union[int, float]]], np.ndarray]]
        2-D axis bounds data consisting of integers or floats, defaults to None.
        Must have a shape of n x 2, where n is the length of ``data``.
    generate_bounds : Optiona[bool]
        Generate bounds for the axis if ``bounds`` is None, by default True.
    attrs : Optional[Dict[str, str]]
        Custom attributes to be added to the generated `xr.DataArray` axis, by
        default None.

        User provided ``attrs`` will be merged with a set of default attributes.
        Default attributes ("axis", "coordinate", "bnds") cannot be overwritten.
        The default "units" attribute is the only default that can be overwritten.

    Returns
    -------
    Tuple[xr.DataArray, Optional[xr.DataArray]]
        A DataArray containing the axis data and optional bounds.

    Raises
    ------
    ValueError
        If ``name`` is not valid CF axis name.

    Examples
    --------
    Create axis and generate bounds (by default):

    >>> lat, bnds = create_axis("lat", np.array([-45, 0, 45]))

    Create axis and bounds from list of floats:

    >>> lat, bnds = create_axis("lat", [-45, 0, 45], bounds=[[-67.5, -22.5], [-22.5, 22.5], [22.5, 67.5]])

    Create axis and disable generating bounds:

    >>> lat, _ = create_axis("lat", np.array([-45, 0, 45]), generate_bounds=False)

    Provide additional attributes and overwrite `units`:

    >>> lat, _ = create_axis(
    >>>     "lat",
    >>>     np.array([-45, 0, 45]),
    >>>     attrs={"generated": str(datetime.date.today()), "units": "degrees_south"},
    >>> )
    """
    bnds = None
    axis_key = None

    if attrs is None:
        attrs = {}

    for cf_axis, names in VAR_NAME_MAP.items():
        if name in names:
            axis_key = cf_axis

            break

    if axis_key is None:
        raise ValueError(f"The name {name!r} is not valid for an axis name.")

    # Replace user attributes with default attributes that can't be overwritten.
    default_axis_attrs = attrs.copy()
    default_axis_attrs.update(COORD_DEFAULT_ATTRS[axis_key].copy())

    # Use the user specified "units" attribute if set.
    try:
        default_axis_attrs["units"] = attrs["units"]
    except KeyError:
        pass

    da = xr.DataArray(data, name=name, dims=[name], attrs=default_axis_attrs)

    if bounds is None:
        if generate_bounds:
            bnds = create_bounds(axis_key, da)
    else:
        bnds = xr.DataArray(bounds, name=f"{name}_bnds", dims=[name, "bnds"])

    if bnds is not None:
        da.attrs["bounds"] = bnds.name

    return da, bnds


def _validate_grid_has_single_axis_dim(
    axis: CFAxisKey, coord_var: Union[xr.DataArray, xr.Dataset]
):
    """Validates that the grid's axis has a single dimension.

    If the grid has multiple dimensions (e.g., "lat" and "latitude" dims), xcdat
    cannot interpret which one to use for grid operations. If ``coord_var`` is
    an ``xr.Dataset``, the grid has multiple dimensions.

    Parameters
    ----------
    axis : CFAxisKey
        The CF axis key ("X", "Y", "T", or "Z").
    coord_var : Union[xr.DataArray, xr.Dataset]
        The dimension coordinate variable(s) for the axis.

    Raises
    ------
    ValueError
        If the grid has multiple dimensions.
    """
    if isinstance(coord_var, xr.Dataset):
        raise ValueError(
            f"Multiple '{axis}' axis dims were found in this dataset, "
            f"{list(coord_var.dims)}. Please drop the unused dimension(s) before "
            "performing grid operations."
        )
