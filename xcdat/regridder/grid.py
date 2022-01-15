from typing import Tuple

import numpy as np
import xarray as xr

# First 50 zeros for the bessel function
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

EPS = 1e-14
ESTIMATE_CONST = 0.25 * (1.0 - np.power(2.0 / np.pi, 2))


def _legendre_polinomial(bessel_zero: int, nlats: int) -> Tuple[float, float, float]:
    # TODO create method for legendre polinomial
    # TODO figure out why this is different compared to numpy.polynomial.legendre, is this wrong?
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


def _bessel_funcation_zeros(n: int):
    values = np.zeros(n)

    lookup_n = min(n, 50)

    values[:lookup_n] = BESSEL_LOOKUP[:lookup_n]

    # interpolate remaining values
    if n > 50:
        for x in range(50, n):
            values[x] = values[x - 1] + np.pi

    return values


def _bessel_zeros(mid: int, nlats: int):
    bessel_zeros = _bessel_funcation_zeros(mid + 1)

    bessel_zeros = np.pad(bessel_zeros, (0, nlats - len(bessel_zeros)))

    weights = np.zeros_like(bessel_zeros)

    for x in range(int(nlats / 2 + 1)):
        zero_poly, first_poly, second_poly = _legendre_polinomial(
            bessel_zeros[x], nlats
        )

        bessel_zeros[x] = zero_poly

        weights[x] = (
            (1.0 - zero_poly * zero_poly)
            * 2.0
            / ((nlats * first_poly) * (nlats * first_poly))
        )

    bessel_zeros[mid if nlats % 2 == 0 else mid + 1 :] = -1.0 * np.flip(
        bessel_zeros[:mid]
    )

    weights[mid if nlats % 2 == 0 else mid + 1 :] = np.flip(weights[:mid])

    if nlats % 2 != 0:
        weights[mid + 1] = 0.0

        mid_weight = 2.0 / np.power(nlats, 2)

        for x in range(2, nlats + 1, 2):
            mid_weight = (mid_weight * np.power(nlats, 2)) / np.power(x - 1, 2)

            weights[mid + 1] = mid_weight

    bessel_zeros = (180.0 / np.pi) * np.arcsin(bessel_zeros)

    return bessel_zeros, weights


def _create_gaussian_axis(nlats: int) -> Tuple[np.ndarray, np.ndarray]:
    mid = int(np.floor(nlats / 2))

    bessel_zeros, weights = _bessel_zeros(mid, nlats)

    bnd_pts = np.zeros((nlats + 1,))
    bnd_pts[0], bnd_pts[nlats] = 1.0, -1.0

    for x in range(1, int(np.floor(nlats / 2)) + 1):
        bnd_pts[x] = bnd_pts[x - 1] - weights[x - 1]

    bnd_pts_mid = int(np.floor(nlats / 2))
    bnd_pts[bnd_pts_mid:] = -1.0 * np.flip(bnd_pts[: bnd_pts_mid + 1])
    bnd_pts = (180.0 / np.pi) * np.arcsin(bnd_pts)

    bnds = np.zeros((bessel_zeros.shape[0], 2))
    bnds[::, 0] = bnd_pts[:-1]
    bnds[::, 1] = bnd_pts[1:]

    return bessel_zeros, bnds


def create_uniform_grid(
    lat_start: float,
    lat_stop: float,
    lat_delta: float,
    lon_start: float,
    lon_stop: float,
    lon_delta: float,
) -> xr.Dataset:
    """
    Creates a uniform rectilinear grid. Sets appropriate attributes
    for lat/lon axis.

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
    grid = xr.Dataset(
        coords={
            "lat": xr.DataArray(
                name="lat",
                data=np.arange(lat_start, lat_stop, lat_delta),
                dims=["lat"],
                attrs={
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "lon": xr.DataArray(
                name="lon",
                data=np.arange(lon_start, lon_stop, lon_delta),
                dims=["lon"],
                attrs={
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
        },
    )

    grid = grid.bounds.add_missing_bounds()

    return grid


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
    lats, bounds = _create_gaussian_axis(nlats)

    grid = xr.Dataset(
        coords={
            "lat": xr.DataArray(
                name="lat",
                data=lats,
                dims=["lat"],
                attrs={
                    "units": "degrees_north",
                    "axis": "Y",
                },
            ),
            "lon": xr.DataArray(
                name="lon",
                data=np.arange(0.0, 360.0, (360.0 / (2 * nlats))),
                dims=["lon"],
                attrs={
                    "units": "degrees_east",
                    "axis": "X",
                },
            ),
        },
    )

    grid = grid.bounds.add_missing_bounds()

    grid["lat_bnds"] = xr.DataArray(
        name="lat_bnds",
        data=bounds,
        coords={"lat": grid.lat},
        dims=["lat", "bnds"],
        attrs={"is_generated": "True"},
    )

    return grid


def create_global_mean_grid(grid: xr.Dataset) -> xr.Dataset:
    """
    Creates a global mean grid.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing the global mean grid.

    """
    lat = grid.cf["lat"]
    lat_bnd = grid.cf.get_bounds("lat")

    out_lat_data = np.array([(lat[0] + lat[-1]) / 2.0])
    out_lat_bnds = np.array([[lat_bnd[0, 0], lat_bnd[-1, 1]]])

    lon = grid.cf["lon"]
    lon_bnd = grid.cf.get_bounds("lon")

    out_lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    out_lon_bnds = np.array([[lon_bnd[0, 0], lon_bnd[-1, 1]]])

    lat_dataarray = xr.DataArray(
        name="lat",
        data=out_lat_data,
        dims=["lat"],
        attrs=grid.cf["lat"].attrs.copy(),
    )

    lon_dataarray = xr.DataArray(
        name="lon",
        data=out_lon_data,
        dims=["lon"],
        attrs=grid.cf["lon"].attrs.copy(),
    )

    return xr.Dataset(
        coords={
            "lat": lat_dataarray,
            "lon": lon_dataarray,
        },
        data_vars={
            "lat_bnds": xr.DataArray(
                name="lat_bnds",
                data=out_lat_bnds,
                dims=["lat", "bnds"],
                coords={
                    "lat": lat_dataarray,
                },
                attrs=grid.cf.get_bounds("lat").attrs.copy(),
            ),
            "lon_bnds": xr.DataArray(
                name="lon_bnds",
                data=out_lon_bnds,
                dims=["lon", "bnds"],
                coords={
                    "lon": lon_dataarray,
                },
                attrs=grid.cf.get_bounds("lon").attrs.copy(),
            ),
        },
    )


def create_zonal_grid(grid: xr.Dataset) -> xr.Dataset:
    """
    Creates a zonal grid.

    Parameters
    ----------
    grid : xr.Dataset
        Source grid.

    Returns
    -------
    xr.Dataset
        A dataset containing a zonal grid.

    """
    lon = grid.cf["lon"]
    lon_bnds = grid.cf.get_bounds("lon")

    out_lon_data = np.array([(lon[0] + lon[-1]) / 2.0])
    out_lon_bnds = np.array([[lon_bnds[0, 0], lon_bnds[-1, 1]]])

    lat_dataarray = xr.DataArray(
        name="lat",
        data=grid.cf["lat"],
        dims=["lat"],
        attrs=grid.cf["lat"].attrs.copy(),
    )

    lon_dataarray = xr.DataArray(
        name="lon",
        data=out_lon_data,
        dims=["lon"],
        attrs=grid.cf["lon"].attrs.copy(),
    )

    return xr.Dataset(
        coords={
            "lat": lat_dataarray,
            "lon": lon_dataarray,
        },
        data_vars={
            "lat_bnds": xr.DataArray(
                name="lat_bnds",
                data=grid.cf.get_bounds("lat"),
                dims=["lat", "bnds"],
                coords={
                    "lat": lat_dataarray,
                },
                attrs=grid.cf.get_bounds("lat").attrs.copy(),
            ),
            "lon_bnds": xr.DataArray(
                name="lon_bnds",
                data=out_lon_bnds,
                dims=["lon", "bnds"],
                coords={
                    "lon": lon_dataarray,
                },
                attrs=grid.cf.get_bounds("lon").attrs.copy(),
            ),
        },
    )
