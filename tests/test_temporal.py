import logging

import cftime
import numpy as np
import pytest
import xarray as xr
from xarray.coding.cftime_offsets import get_date_type
from xarray.tests import requires_dask

from tests.fixtures import generate_dataset
from xcdat._logger import _setup_custom_logger
from xcdat.temporal import (
    TemporalAccessor,
    _contains_datetime_like_objects,
    _get_datetime_like_type,
)

logger = _setup_custom_logger("xcdat.temporal", propagate=True)


class TestTemporalAccessor:
    def test__init__(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        obj = TemporalAccessor(ds)
        assert obj._dataset.identical(ds)

    def test_decorator(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        obj = ds.temporal
        assert obj._dataset.identical(ds)


class TestAverage:
    def test_raises_error_if_time_coords_are_not_decoded(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=False, cf_compliant=False, has_bounds=True
        )

        with pytest.raises(TypeError):
            ds.temporal.average("ts")

    def test_defaults_calendar_attribute_to_standard_if_missing(self, caplog):
        # Silence warning to not pollute test suite output
        caplog.set_level(logging.CRITICAL)

        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        ds.ts.time.encoding = {}

        ds.temporal.average("ts")

        assert ds.temporal.calendar == "standard"

    def test_averages_for_yearly_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                            "2002-01-01T00:00:00.000000000",
                            "2003-01-01T00:00:00.000000000",
                            "2004-01-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            }
        )
        ds.time.encoding = {"calendar": "standard"}
        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2001-01-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2002-01-01T00:00:00.000000000"],
                    ["2002-01-01T00:00:00.000000000", "2003-01-01T00:00:00.000000000"],
                    ["2003-01-01T00:00:00.000000000", "2004-01-01T00:00:00.000000000"],
                    ["2004-01-01T00:00:00.000000000", "2005-01-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[np.nan]], [[1]], [[1]], [[2]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by year
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.5]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "year",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.5]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "year",
                "weighted": "False",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_averages_for_monthly_time_series(self):
        # Set up dataset
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-02-01T00:00:00.000000000",
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            }
        )
        ds.time.encoding = {"calendar": "standard"}

        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-04-01T00:00:00.000000000", "2000-05-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[np.nan]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by month
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.24362357]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "month",
                "weighted": "True",
            },
        )

        xr.testing.assert_allclose(result, expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.25]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "month",
                "weighted": "False",
            },
        )
        xr.testing.assert_allclose(result, expected)

    def test_averages_for_daily_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T00:00:00.000000000",
                            "2000-01-02T00:00:00.000000000",
                            "2000-01-03T00:00:00.000000000",
                            "2000-01-04T00:00:00.000000000",
                            "2000-01-05T00:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            }
        )
        ds.time.encoding = {"calendar": "standard"}

        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-01-02T00:00:00.000000000"],
                    ["2000-01-02T00:00:00.000000000", "2000-01-03T00:00:00.000000000"],
                    ["2000-01-03T00:00:00.000000000", "2000-01-04T00:00:00.000000000"],
                    ["2000-01-04T00:00:00.000000000", "2000-01-05T00:00:00.000000000"],
                    ["2000-01-05T00:00:00.000000000", "2000-01-06T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by day
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "day",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "day",
                "weighted": "False",
            },
        )
        xr.testing.assert_identical(result, expected)

    def test_averages_for_hourly_time_series(self):
        ds = xr.Dataset(
            coords={
                "lat": [-90],
                "lon": [0],
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-01T01:00:00.000000000",
                            "2000-01-01T02:00:00.000000000",
                            "2000-01-01T03:00:00.000000000",
                            "2000-01-01T04:00:00.000000000",
                            "2000-01-01T05:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            }
        )
        ds.time.encoding = {"calendar": "standard"}

        ds["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T01:00:00.000000000", "2000-01-01T02:00:00.000000000"],
                    ["2000-01-01T02:00:00.000000000", "2000-01-01T03:00:00.000000000"],
                    ["2000-01-01T03:00:00.000000000", "2000-01-01T04:00:00.000000000"],
                    ["2000-01-01T04:00:00.000000000", "2000-01-01T05:00:00.000000000"],
                    ["2000-01-01T05:00:00.000000000", "2000-01-01T06:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        ds["ts"] = xr.DataArray(
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[1]]]),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["time", "lat", "lon"],
        )

        # Test averages weighted by hour
        result = ds.temporal.average("ts")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "hour",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

        # Test unweighted averages
        result = ds.temporal.average("ts", weighted=False)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            data=np.array([[1.2]]),
            coords={"lat": expected.lat, "lon": expected.lon},
            dims=["lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "average",
                "freq": "hour",
                "weighted": "False",
            },
        )

        xr.testing.assert_identical(result, expected)


class TestGroupAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-03-16T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-02-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        time.encoding = {"calendar": "standard"}
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        self.ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        self.ds.time.attrs["bounds"] = "time_bnds"

        self.ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
            attrs={"test_attr": "test"},
        )

    def test_raises_error_if_time_coords_are_not_decoded(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=False, cf_compliant=False, has_bounds=True
        )

        with pytest.raises(TypeError):
            ds.temporal.group_average("ts", freq="year")

    def test_defaults_calendar_attribute_to_standard_if_missing(self, caplog):
        # Silence warning to not pollute test suite output
        caplog.set_level(logging.CRITICAL)

        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        ds.ts.time.encoding = {}

        ds.temporal.group_average("ts", freq="year")

        assert ds.temporal.calendar == "standard"

    def test_weighted_annual_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "year")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.25409836]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    coords={
                        "time": np.array(
                            [
                                cftime.DatetimeGregorian(2000, 1, 1),
                                cftime.DatetimeGregorian(2001, 1, 1),
                            ],
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "year",
                "weighted": "True",
            },
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs
        assert result.time.attrs == expected.time.attrs

    def test_weighted_annual_averages_and_skipna(self):
        ds = self.ds.copy(deep=True)
        ds.ts[0] = np.nan

        result = ds.temporal.group_average("ts", "year", skipna=True)
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    coords={
                        "time": np.array(
                            [
                                cftime.DatetimeGregorian(2000, 1, 1),
                                cftime.DatetimeGregorian(2001, 1, 1),
                            ],
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "year",
                "weighted": "True",
            },
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs
        assert result.time.attrs == expected.time.attrs

    @requires_dask
    def test_weighted_annual_averages_with_chunking(self):
        ds = self.ds.copy().chunk({"time": 2})

        result = ds.temporal.group_average("ts", "year")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.25409836]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    coords={
                        "time": np.array(
                            [
                                cftime.DatetimeGregorian(2000, 1, 1),
                                cftime.DatetimeGregorian(2001, 1, 1),
                            ],
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "year",
                "weighted": "True",
            },
        )

        xr.testing.assert_allclose(result, expected)
        assert result.ts.attrs == expected.ts.attrs
        assert result.time.attrs == expected.time.attrs

    def test_weighted_seasonal_averages_with_DJF_and_drop_incomplete_seasons(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )
        expected = ds.copy()
        # Drop the incomplete DJF seasons
        expected = expected.isel(time=slice(2, -1))
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1]], [[1]], [[1]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_averages_with_DJF_without_dropping_incomplete_seasons(
        self,
    ):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": False},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "False",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_averages_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"dec_mode": "JFD"},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    coords={
                        "time": np.array(
                            [
                                cftime.DatetimeGregorian(2000, 1, 1),
                                cftime.DatetimeGregorian(2000, 4, 1),
                                cftime.DatetimeGregorian(2000, 7, 1),
                                cftime.DatetimeGregorian(2000, 10, 1),
                                cftime.DatetimeGregorian(2001, 1, 1),
                            ],
                        )
                    },
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "JFD",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_custom_seasonal_averages(self):
        ds = self.ds.copy()

        custom_seasons = [
            ["Jan", "Feb", "Mar"],
            ["Apr", "May", "Jun"],
            ["Jul", "Aug", "Sep"],
            ["Oct", "Nov", "Dec"],
        ]
        result = ds.temporal.group_average(
            "ts",
            "season",
            season_config={"custom_seasons": custom_seasons},
        )
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[1.5]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 2, 1),
                            cftime.DatetimeGregorian(2000, 5, 1),
                            cftime.DatetimeGregorian(2000, 8, 1),
                            cftime.DatetimeGregorian(2001, 2, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "season",
                "custom_seasons": [
                    "JanFebMar",
                    "AprMayJun",
                    "JulAugSep",
                    "OctNovDec",
                ],
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_raises_error_with_incorrect_custom_seasons_argument(self):
        # Test raises error with non-3 letter strings
        with pytest.raises(ValueError):
            custom_seasons = [
                ["J", "Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

        # Test raises error with missing month(s)
        with pytest.raises(ValueError):
            custom_seasons = [
                ["Feb", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

        # Test raises error if duplicate month(s) were found
        with pytest.raises(ValueError):
            custom_seasons = [
                ["Jan", "Jan", "Mar"],
                ["Apr", "May", "Jun"],
                ["Jul", "Aug", "Sep"],
                ["Oct", "Nov", "Dec"],
            ]
            self.ds.temporal.group_average(
                "ts",
                "season",
                season_config={"custom_seasons": custom_seasons},
            )

    def test_weighted_monthly_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "month")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2000, 3, 1),
                            cftime.DatetimeGregorian(2000, 6, 1),
                            cftime.DatetimeGregorian(2000, 9, 1),
                            cftime.DatetimeGregorian(2001, 2, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "month",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_monthly_averages_with_masked_data(self):
        ds = self.ds.copy()
        ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[np.nan]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
            attrs={"test_attr": "test"},
        )

        result = ds.temporal.group_average("ts", "month")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[np.nan]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2000, 3, 1),
                            cftime.DatetimeGregorian(2000, 6, 1),
                            cftime.DatetimeGregorian(2000, 9, 1),
                            cftime.DatetimeGregorian(2001, 2, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "month",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_daily_averages(self):
        ds = self.ds.copy()

        result = ds.temporal.group_average("ts", "day")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 16),
                            cftime.DatetimeGregorian(2000, 3, 16),
                            cftime.DatetimeGregorian(2000, 6, 16),
                            cftime.DatetimeGregorian(2000, 9, 16),
                            cftime.DatetimeGregorian(2001, 2, 15),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "day",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_hourly_averages(self):
        ds = self.ds.copy()
        ds.coords["time"].attrs["bounds"] = "time_bnds"

        result = ds.temporal.group_average("ts", "hour")
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[2]], [[1]], [[1]], [[1]], [[2]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 16, 12),
                            cftime.DatetimeGregorian(2000, 3, 16, 12),
                            cftime.DatetimeGregorian(2000, 6, 16, 0),
                            cftime.DatetimeGregorian(2000, 9, 16, 0),
                            cftime.DatetimeGregorian(2001, 2, 15, 12),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "group_average",
                "freq": "hour",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)


class TestClimatology:
    # TODO: Update TestClimatology tests to use other numbers rather than 1's
    # for better test reliability and accuracy. This may require subsetting.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_if_time_coords_are_not_decoded(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=False, cf_compliant=False, has_bounds=True
        )

        with pytest.raises(TypeError):
            ds.temporal.climatology("ts", freq="year")

    def test_defaults_calendar_attribute_to_standard_if_missing(self, caplog):
        # Silence warning to not pollute test suite output
        caplog.set_level(logging.CRITICAL)

        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        ds.ts.time.encoding = {}

        ds.temporal.climatology("ts", freq="season")

        assert ds.temporal.calendar == "standard"

    def test_raises_error_if_reference_period_arg_is_incorrect(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            ds.temporal.climatology(
                "ts",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
                reference_period=("01-01-2000", "01-01-2000"),
            )

        with pytest.raises(ValueError):
            ds.temporal.climatology(
                "ts",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
                reference_period=("01-01-2000"),
            )

    def test_subsets_climatology_based_on_reference_period(self):
        ds = self.ds.copy()

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            reference_period=("2000-01-01", "2000-06-01"),
        )

        # The first month Jan/Feb are dropped (incomplete DJF). This means
        # only the MAM season will be present, with April being the middle month
        # (represented by month number 4).
        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array([cftime.DatetimeGregorian(1, 4, 1)]),
            coords={
                "time": np.array([cftime.DatetimeGregorian(1, 4, 1)]),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((1, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_climatology_with_DJF(self):
        ds = self.ds.copy()

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_climatology_with_DJF_and_skipna(self):
        ds = self.ds.copy(deep=True)

        # Replace all MAM values with np.nan.
        djf_months = [3, 4, 5]
        for mon in djf_months:
            ds["ts"] = ds.ts.where(ds.ts.time.dt.month != mon, np.nan)

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            skipna=True,
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )
        expected.ts[1] = np.nan

        # MAM should be np.nan
        assert result.identical(expected)

    @requires_dask
    def test_chunked_weighted_seasonal_climatology_with_DJF(self):
        ds = self.ds.copy().chunk({"time": 2})

        result = ds.temporal.climatology(
            "ts",
            "season",
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_climatology_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.climatology(
            "ts", "season", season_config={"dec_mode": "JFD"}
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "JFD",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_custom_seasonal_climatology(self):
        ds = self.ds.copy()

        custom_seasons = [
            ["Jan", "Feb", "Mar"],
            ["Apr", "May", "Jun"],
            ["Jul", "Aug", "Sep"],
            ["Oct", "Nov", "Dec"],
        ]
        result = ds.temporal.climatology(
            "ts", "season", season_config={"custom_seasons": custom_seasons}
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 2, 1),
                    cftime.DatetimeGregorian(1, 5, 1),
                    cftime.DatetimeGregorian(1, 8, 1),
                    cftime.DatetimeGregorian(1, 11, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 5, 1),
                        cftime.DatetimeGregorian(1, 8, 1),
                        cftime.DatetimeGregorian(1, 11, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )

        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((4, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "season",
                "weighted": "True",
                "custom_seasons": [
                    "JanFebMar",
                    "AprMayJun",
                    "JulAugSep",
                    "OctNovDec",
                ],
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_monthly_climatology(self):
        result = self.ds.temporal.climatology("ts", "month")

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 2, 1),
                    cftime.DatetimeGregorian(1, 3, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 5, 1),
                    cftime.DatetimeGregorian(1, 6, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 8, 1),
                    cftime.DatetimeGregorian(1, 9, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                    cftime.DatetimeGregorian(1, 11, 1),
                    cftime.DatetimeGregorian(1, 12, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 3, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 5, 1),
                        cftime.DatetimeGregorian(1, 6, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 8, 1),
                        cftime.DatetimeGregorian(1, 9, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 11, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )

        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "month",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_unweighted_monthly_climatology(self):
        result = self.ds.temporal.climatology("ts", "month", weighted=False)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 1),
                    cftime.DatetimeGregorian(1, 2, 1),
                    cftime.DatetimeGregorian(1, 3, 1),
                    cftime.DatetimeGregorian(1, 4, 1),
                    cftime.DatetimeGregorian(1, 5, 1),
                    cftime.DatetimeGregorian(1, 6, 1),
                    cftime.DatetimeGregorian(1, 7, 1),
                    cftime.DatetimeGregorian(1, 8, 1),
                    cftime.DatetimeGregorian(1, 9, 1),
                    cftime.DatetimeGregorian(1, 10, 1),
                    cftime.DatetimeGregorian(1, 11, 1),
                    cftime.DatetimeGregorian(1, 12, 1),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 3, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 5, 1),
                        cftime.DatetimeGregorian(1, 6, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 8, 1),
                        cftime.DatetimeGregorian(1, 9, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 11, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "month",
                "weighted": "False",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_daily_climatology(self):
        result = self.ds.temporal.climatology("ts", "day", weighted=True)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 16),
                    cftime.DatetimeGregorian(1, 2, 15),
                    cftime.DatetimeGregorian(1, 3, 16),
                    cftime.DatetimeGregorian(1, 4, 16),
                    cftime.DatetimeGregorian(1, 5, 16),
                    cftime.DatetimeGregorian(1, 6, 16),
                    cftime.DatetimeGregorian(1, 7, 16),
                    cftime.DatetimeGregorian(1, 8, 16),
                    cftime.DatetimeGregorian(1, 9, 16),
                    cftime.DatetimeGregorian(1, 10, 16),
                    cftime.DatetimeGregorian(1, 11, 16),
                    cftime.DatetimeGregorian(1, 12, 16),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 16),
                        cftime.DatetimeGregorian(1, 2, 15),
                        cftime.DatetimeGregorian(1, 3, 16),
                        cftime.DatetimeGregorian(1, 4, 16),
                        cftime.DatetimeGregorian(1, 5, 16),
                        cftime.DatetimeGregorian(1, 6, 16),
                        cftime.DatetimeGregorian(1, 7, 16),
                        cftime.DatetimeGregorian(1, 8, 16),
                        cftime.DatetimeGregorian(1, 9, 16),
                        cftime.DatetimeGregorian(1, 10, 16),
                        cftime.DatetimeGregorian(1, 11, 16),
                        cftime.DatetimeGregorian(1, 12, 16),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "day",
                "weighted": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_daily_climatology_drops_leap_days_with_matching_calendar(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-02-29T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-03-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-03-01T00:00:00.000000000", "2001-04-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        ds.time.attrs["bounds"] = "time_bnds"
        ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
            dims=["time", "lat", "lon"],
            attrs={"test_attr": "test"},
        )

        # Loop over calendars and test results.
        calendars = ["gregorian", "proleptic_gregorian", "standard"]
        for calendar in calendars:
            ds_new = ds.copy()
            ds_new.time.encoding = {"calendar": calendar}

            result = ds_new.temporal.climatology("ts", "day", weighted=True)
            expected = ds.copy()
            expected = expected.drop_dims("time")
            expected["ts"] = xr.DataArray(
                name="ts",
                data=np.array([[[2.0]], [[2.0]], [[1.0]], [[1.0]]]),
                coords={
                    "lat": expected.lat,
                    "lon": expected.lon,
                    "time": xr.DataArray(
                        data=np.array(
                            [
                                get_date_type(calendar)(1, 1, 16),
                                get_date_type(calendar)(1, 3, 15),
                                get_date_type(calendar)(1, 6, 16),
                                get_date_type(calendar)(1, 9, 16),
                            ],
                        ),
                        dims=["time"],
                        attrs={
                            "axis": "T",
                            "long_name": "time",
                            "standard_name": "time",
                            "bounds": "time_bnds",
                        },
                    ),
                },
                dims=["time", "lat", "lon"],
                attrs={
                    "test_attr": "test",
                    "operation": "temporal_avg",
                    "mode": "climatology",
                    "freq": "day",
                    "weighted": "True",
                },
            )

            xr.testing.assert_identical(result, expected)

    def test_unweighted_daily_climatology(self):
        result = self.ds.temporal.climatology("ts", "day", weighted=False)

        expected = self.ds.copy()
        expected = expected.drop_dims("time")
        expected_time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(1, 1, 16),
                    cftime.DatetimeGregorian(1, 2, 15),
                    cftime.DatetimeGregorian(1, 3, 16),
                    cftime.DatetimeGregorian(1, 4, 16),
                    cftime.DatetimeGregorian(1, 5, 16),
                    cftime.DatetimeGregorian(1, 6, 16),
                    cftime.DatetimeGregorian(1, 7, 16),
                    cftime.DatetimeGregorian(1, 8, 16),
                    cftime.DatetimeGregorian(1, 9, 16),
                    cftime.DatetimeGregorian(1, 10, 16),
                    cftime.DatetimeGregorian(1, 11, 16),
                    cftime.DatetimeGregorian(1, 12, 16),
                ],
            ),
            coords={
                "time": np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 16),
                        cftime.DatetimeGregorian(1, 2, 15),
                        cftime.DatetimeGregorian(1, 3, 16),
                        cftime.DatetimeGregorian(1, 4, 16),
                        cftime.DatetimeGregorian(1, 5, 16),
                        cftime.DatetimeGregorian(1, 6, 16),
                        cftime.DatetimeGregorian(1, 7, 16),
                        cftime.DatetimeGregorian(1, 8, 16),
                        cftime.DatetimeGregorian(1, 9, 16),
                        cftime.DatetimeGregorian(1, 10, 16),
                        cftime.DatetimeGregorian(1, 11, 16),
                        cftime.DatetimeGregorian(1, 12, 16),
                    ],
                ),
            },
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
                "bounds": "time_bnds",
            },
        )
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.ones((12, 4, 4)),
            coords={"lat": expected.lat, "lon": expected.lon, "time": expected_time},
            dims=["time", "lat", "lon"],
            attrs={
                "operation": "temporal_avg",
                "mode": "climatology",
                "freq": "day",
                "weighted": "False",
            },
        )

        xr.testing.assert_identical(result, expected)


class TestDepartures:
    @pytest.fixture(autouse=True)
    def setup(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-03-16T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-02-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        time.encoding = {"calendar": "standard"}
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-03-01T00:00:00.000000000", "2000-04-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-02-01T00:00:00.000000000", "2001-03-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        self.ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        self.ds.time.attrs["bounds"] = "time_bnds"

        self.ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": self.ds.time, "lat": self.ds.lat, "lon": self.ds.lon},
            dims=["time", "lat", "lon"],
            attrs={"test_attr": "test"},
        )

    def test_raises_error_if_time_coords_are_not_decoded(self):
        ds: xr.Dataset = generate_dataset(
            decode_times=False, cf_compliant=False, has_bounds=True
        )

        with pytest.raises(TypeError):
            ds.temporal.departures("ts", freq="season")

    def test_defaults_calendar_attribute_to_standard_if_missing(self, caplog):
        # Silence warning to not pollute test suite output
        caplog.set_level(logging.CRITICAL)

        ds: xr.Dataset = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        ds.ts.time.encoding = {}

        ds.temporal.departures("ts", freq="season")

        assert ds.temporal.calendar == "standard"

    def test_raises_error_if_reference_period_arg_is_incorrect(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            ds.temporal.departures(
                "ts",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
                reference_period=("01-01-2000", "01-01-2000"),
            )

        with pytest.raises(ValueError):
            ds.temporal.departures(
                "ts",
                "season",
                season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
                reference_period=("01-01-2000"),
            )

    def test_seasonal_departures_relative_to_climatology_reference_period(self):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=True,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            reference_period=("2000-01-01", "2000-06-01"),
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[np.nan]], [[np.nan]], [[np.nan]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_monthly_departures_relative_to_climatology_reference_period_with_same_output_freq(
        self,
    ):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "month",
            weighted=True,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            reference_period=("2000-01-01", "2000-06-01"),
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[0.0]], [[np.nan]], [[np.nan]], [[np.nan]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            "2000-01-16T12:00:00.000000000",
                            "2000-03-16T12:00:00.000000000",
                            "2000-06-16T00:00:00.000000000",
                            "2000-09-16T00:00:00.000000000",
                            "2001-02-15T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "month",
                "weighted": "True",
            },
        )
        expected["time_bnds"] = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                    ],
                    [
                        "2000-03-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                    ],
                    [
                        "2000-06-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                    ],
                    [
                        "2000-09-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                    ],
                    [
                        "2001-02-01T00:00:00.000000000",
                        "2001-03-01T00:00:00.000000000",
                    ],
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time", "bnds"],
            attrs={
                "xcdat_bounds": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_departures_with_DJF(self):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=True,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_seasonal_departures_with_DJF_and_skipna(self):
        ds = self.ds.copy(deep=True)

        # Replace all MAM values with np.nan.
        djf_months = [3, 4, 5]
        for mon in djf_months:
            ds["ts"] = ds.ts.where(ds.ts.time.dt.month != mon, np.nan)

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=True,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
            skipna=True,
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[np.nan]], [[0.0]], [[0.0]], [[0.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        assert result.identical(expected)

    def test_weighted_seasonal_departures_with_DJF_and_keep_weights(self):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=True,
            keep_weights=True,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "True",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )
        expected["time_wts"] = xr.DataArray(
            name="ts",
            data=np.array([1.0, 1.0, 1.0, 1.0]),
            coords={
                "time_original": xr.DataArray(
                    data=np.array(
                        [
                            "2000-03-16T12:00:00.000000000",
                            "2000-06-16T00:00:00.000000000",
                            "2000-09-16T00:00:00.000000000",
                            "2001-02-15T12:00:00.000000000",
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=["time_original"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                )
            },
            dims=["time_original"],
        )

        xr.testing.assert_identical(result, expected)

    def test_unweighted_seasonal_departures_with_DJF(self):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=False,
            season_config={"dec_mode": "DJF", "drop_incomplete_djf": True},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "dec_mode": "DJF",
                "drop_incomplete_djf": "True",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_unweighted_seasonal_departures_with_JFD(self):
        ds = self.ds.copy()

        result = ds.temporal.departures(
            "ts",
            "season",
            weighted=False,
            season_config={"dec_mode": "JFD"},
        )

        expected = ds.copy()
        expected = expected.drop_dims("time")
        expected["ts"] = xr.DataArray(
            name="ts",
            data=np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]),
            coords={
                "lat": expected.lat,
                "lon": expected.lon,
                "time": xr.DataArray(
                    data=np.array(
                        [
                            cftime.DatetimeGregorian(2000, 1, 1),
                            cftime.DatetimeGregorian(2000, 4, 1),
                            cftime.DatetimeGregorian(2000, 7, 1),
                            cftime.DatetimeGregorian(2000, 10, 1),
                            cftime.DatetimeGregorian(2001, 1, 1),
                        ],
                    ),
                    dims=["time"],
                    attrs={
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "bounds": "time_bnds",
                    },
                ),
            },
            dims=["time", "lat", "lon"],
            attrs={
                "test_attr": "test",
                "operation": "temporal_avg",
                "mode": "departures",
                "freq": "season",
                "weighted": "False",
                "dec_mode": "JFD",
            },
        )

        xr.testing.assert_identical(result, expected)

    def test_weighted_daily_departures_drops_leap_days_with_matching_calendar(self):
        time = xr.DataArray(
            data=np.array(
                [
                    "2000-01-16T12:00:00.000000000",
                    "2000-02-29T12:00:00.000000000",
                    "2000-06-16T00:00:00.000000000",
                    "2000-09-16T00:00:00.000000000",
                    "2001-03-15T12:00:00.000000000",
                ],
                dtype="datetime64[ns]",
            ),
            dims=["time"],
            attrs={"axis": "T", "long_name": "time", "standard_name": "time"},
        )
        time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T00:00:00.000000000", "2000-02-01T00:00:00.000000000"],
                    ["2000-02-01T00:00:00.000000000", "2000-03-01T00:00:00.000000000"],
                    ["2000-06-01T00:00:00.000000000", "2000-07-01T00:00:00.000000000"],
                    ["2000-09-01T00:00:00.000000000", "2000-10-01T00:00:00.000000000"],
                    ["2001-03-01T00:00:00.000000000", "2001-04-01T00:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": time},
            dims=["time", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        ds = xr.Dataset(
            data_vars={"time_bnds": time_bnds},
            coords={"lat": [-90], "lon": [0], "time": time},
        )
        ds.time.attrs["bounds"] = "time_bnds"
        ds["ts"] = xr.DataArray(
            data=np.array(
                [[[2.0]], [[1.0]], [[1.0]], [[1.0]], [[2.0]]], dtype="float64"
            ),
            coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
            dims=["time", "lat", "lon"],
            attrs={"test_attr": "test"},
        )

        # Loop over calendars and test results.
        calendars = ["gregorian", "proleptic_gregorian", "standard"]
        for calendar in calendars:
            ds_new = ds.copy()
            ds_new.time.encoding = {"calendar": calendar}

            result = ds_new.temporal.departures("ts", "day", weighted=True)
            expected = xr.Dataset(
                coords={
                    "lat": ds.lat,
                    "lon": ds.lon,
                    "time": xr.DataArray(
                        data=np.array(
                            [
                                cftime.DatetimeGregorian(2000, 1, 16),
                                cftime.DatetimeGregorian(2000, 6, 16),
                                cftime.DatetimeGregorian(2000, 9, 16),
                                cftime.DatetimeGregorian(2001, 3, 15),
                            ],
                        ),
                        dims=["time"],
                        attrs={
                            "axis": "T",
                            "long_name": "time",
                            "standard_name": "time",
                            "bounds": "time_bnds",
                        },
                    ),
                },
                data_vars={
                    "ts": xr.DataArray(
                        name="ts",
                        data=np.array([[[0]], [[0]], [[0]], [[0]]]),
                        dims=["time", "lat", "lon"],
                        attrs={
                            "test_attr": "test",
                            "operation": "temporal_avg",
                            "mode": "departures",
                            "freq": "day",
                            "weighted": "True",
                        },
                    ),
                },
            )

            xr.testing.assert_identical(result, expected)


def _check_each_weight_group_adds_up_to_1(ds: xr.Dataset, weights: xr.DataArray):
    """Check that the sum of the weights in each group adds up to 1.0 (or 100%).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with the temporal accessor class attached.
    weights : xr.DataArray
        The weights to check, produced by the `_get_weights` method.
    """
    time_lengths = ds.time_bnds[:, 1] - ds.time_bnds[:, 0]
    time_lengths = time_lengths.astype(np.float64)

    grouped_time_lengths = ds.temporal._group_data(time_lengths)

    actual_sum = ds.temporal._group_data(weights).sum().values
    expected_sum = np.ones(len(grouped_time_lengths.groups))
    np.testing.assert_allclose(actual_sum, expected_sum)


class Test_GetWeights:
    class TestWeightsForAverageMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(
                decode_times=True, cf_compliant=False, has_bounds=True
            )

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "average"
            ds.temporal._freq = "year"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.08469945,
                    0.07923497,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_monthly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "average"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="month",
                data=np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 3, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 5, 1),
                        cftime.DatetimeGregorian(1, 6, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 8, 1),
                        cftime.DatetimeGregorian(1, 9, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 11, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.5,
                    0.50877193,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    # This is a leap year month, so the weight is less.
                    0.49122807,
                    0.5,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

    class TestWeightsForGroupAverageMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(
                decode_times=True, cf_compliant=False, has_bounds=True
            )

        def test_weights_for_yearly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "year"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.08469945,
                    0.07923497,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.08196721,
                    0.08469945,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_monthly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month",
                data=np.array(
                    [
                        "2000-01-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        "2000-03-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        "2000-06-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        "2000-09-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        "2000-12-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-02-01T00:00:00.000000000",
                        "2001-12-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.ones(15)
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_seasonal_averages_with_DJF_and_drop_incomplete_seasons(
            self,
        ):
            ds = self.ds.copy()

            # Replace time and time bounds with incomplete seasons removed
            ds = ds.drop_dims("time")
            ds.coords["time"] = xr.DataArray(
                data=np.array(
                    [
                        "2000-03-16T12:00:00.000000000",
                        "2000-04-16T00:00:00.000000000",
                        "2000-05-16T12:00:00.000000000",
                        "2000-06-16T00:00:00.000000000",
                        "2000-07-16T12:00:00.000000000",
                        "2000-08-16T12:00:00.000000000",
                        "2000-09-16T00:00:00.000000000",
                        "2000-10-16T12:00:00.000000000",
                        "2000-11-16T00:00:00.000000000",
                        "2000-12-16T12:00:00.000000000",
                        "2001-01-16T12:00:00.000000000",
                        "2001-02-15T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            ds.time.encoding = {"calendar": "standard"}
            ds["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((12, 4, 4)),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                dims=["time", "lat", "lon"],
            )
            ds["time_bnds"] = xr.DataArray(
                name="time_bnds",
                data=np.array(
                    [
                        [
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                        ],
                        [
                            "2000-04-01T00:00:00.000000000",
                            "2000-05-01T00:00:00.000000000",
                        ],
                        [
                            "2000-05-01T00:00:00.000000000",
                            "2000-06-01T00:00:00.000000000",
                        ],
                        [
                            "2000-06-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                        ],
                        [
                            "2000-07-01T00:00:00.000000000",
                            "2000-08-01T00:00:00.000000000",
                        ],
                        [
                            "2000-08-01T00:00:00.000000000",
                            "2000-09-01T00:00:00.000000000",
                        ],
                        [
                            "2000-09-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                        ],
                        [
                            "2000-10-01T00:00:00.000000000",
                            "2000-11-01T00:00:00.000000000",
                        ],
                        [
                            "2000-11-01T00:00:00.000000000",
                            "2000-12-01T00:00:00.000000000",
                        ],
                        [
                            "2000-12-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        [
                            "2001-01-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        [
                            "2001-02-01T00:00:00.000000000",
                            "2001-03-01T00:00:00.000000000",
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time", "bnds"],
                attrs={
                    "xcdat_bounds": "True",
                },
            )

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "DJF"}
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 MAM
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        # 2000 JJA
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        # 2000 SON
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        # 2001 DJF
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34444444,
                    0.34444444,
                    0.31111111,
                ]
            )
            assert np.allclose(result, expected, equal_nan=True)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_seasonal_averages_with_JFD(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 JFD
                        "2000-01-01T00:00:00.000000000",
                        "2000-01-01T00:00:00.000000000",
                        # 2000 MAM
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        "2000-04-01T00:00:00.000000000",
                        # 2000 JJA
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        "2000-07-01T00:00:00.000000000",
                        # 2000 SON
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        "2000-10-01T00:00:00.000000000",
                        # 2000 JFD
                        "2000-01-01T00:00:00.000000000",
                        # 2001 JFD
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                        "2001-01-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.34065934,
                    0.31868132,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34065934,
                    0.34444444,
                    0.31111111,
                    0.34444444,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_custom_season_time_series_weights(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._season_config = {
                "custom_seasons": {
                    "JanFebMar": ["Jan", "Feb", "Mar"],
                    "AprMayJun": ["Apr", "May", "Jun"],
                    "JulAugSep": ["Jul", "Aug", "Sep"],
                    "OctNovDec": ["Oct", "Nov", "Dec"],
                }
            }

            ds.temporal._labeled_time = xr.DataArray(
                name="year_season",
                data=np.array(
                    [
                        # 2000 JanFebMar
                        "2000-02-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        "2000-02-01T00:00:00.000000000",
                        # 2000 AprMayJun
                        "2000-05-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        "2000-05-01T00:00:00.000000000",
                        # 2000 JunAugSep
                        "2000-08-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        "2000-08-01T00:00:00.000000000",
                        # 2000 OctNovDec
                        "2000-11-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        "2000-11-01T00:00:00.000000000",
                        # 2001 JanFebMar
                        "2001-02-01T00:00:00.000000000",
                        "2001-02-01T00:00:00.000000000",
                        "2002-02-01T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.34065934,
                    0.31868132,
                    0.34065934,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.33695652,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.52542373,
                    0.47457627,
                    1.0,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_daily_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "day"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month_day",
                data=np.array(
                    [
                        "2000-01-16T00:00:00.000000000",
                        "2000-02-15T00:00:00.000000000",
                        "2000-03-16T00:00:00.000000000",
                        "2000-04-16T00:00:00.000000000",
                        "2000-05-16T00:00:00.000000000",
                        "2000-06-16T00:00:00.000000000",
                        "2000-07-16T00:00:00.000000000",
                        "2000-08-16T00:00:00.000000000",
                        "2000-09-16T00:00:00.000000000",
                        "2000-10-16T00:00:00.000000000",
                        "2000-11-16T00:00:00.000000000",
                        "2000-12-16T00:00:00.000000000",
                        "2001-01-16T00:00:00.000000000",
                        "2001-02-15T00:00:00.000000000",
                        "2001-12-16T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.ones(15)
            assert np.allclose(result, expected)

        def test_weights_for_hourly_averages(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "group_average"
            ds.temporal._freq = "hour"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="year_month_day_hour",
                data=np.array(
                    [
                        "2000-01-16T12:00:00.000000000",
                        "2000-02-15T12:00:00.000000000",
                        "2000-03-16T12:00:00.000000000",
                        "2000-04-16T00:00:00.000000000",
                        "2000-05-16T12:00:00.000000000",
                        "2000-06-16T00:00:00.000000000",
                        "2000-07-16T12:00:00.000000000",
                        "2000-08-16T12:00:00.000000000",
                        "2000-09-16T00:00:00.000000000",
                        "2000-10-16T12:00:00.000000000",
                        "2000-11-16T00:00:00.000000000",
                        "2000-12-16T12:00:00.000000000",
                        "2001-01-16T12:00:00.000000000",
                        "2001-02-15T00:00:00.000000000",
                        "2001-12-16T12:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.ones(15)
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

    class TestWeightsForClimatologyMode:
        @pytest.fixture(autouse=True)
        def setup(self):
            self.ds: xr.Dataset = generate_dataset(
                decode_times=True, cf_compliant=False, has_bounds=True
            )

        def test_weights_for_seasonal_climatology_with_DJF(self):
            ds = self.ds.copy()

            # Replace time and time bounds with incomplete seasons removed
            ds = ds.drop_dims("time")
            ds.coords["time"] = xr.DataArray(
                data=np.array(
                    [
                        "2000-03-16T12:00:00.000000000",
                        "2000-04-16T00:00:00.000000000",
                        "2000-05-16T12:00:00.000000000",
                        "2000-06-16T00:00:00.000000000",
                        "2000-07-16T12:00:00.000000000",
                        "2000-08-16T12:00:00.000000000",
                        "2000-09-16T00:00:00.000000000",
                        "2000-10-16T12:00:00.000000000",
                        "2000-11-16T00:00:00.000000000",
                        "2000-12-16T12:00:00.000000000",
                        "2001-01-16T12:00:00.000000000",
                        "2001-02-15T00:00:00.000000000",
                    ],
                    dtype="datetime64[ns]",
                ),
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            ds.time.encoding = {"calendar": "standard"}
            ds["ts"] = xr.DataArray(
                name="ts",
                data=np.ones((12, 4, 4)),
                coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
                dims=["time", "lat", "lon"],
            )
            ds["time_bnds"] = xr.DataArray(
                name="time_bnds",
                data=np.array(
                    [
                        [
                            "2000-03-01T00:00:00.000000000",
                            "2000-04-01T00:00:00.000000000",
                        ],
                        [
                            "2000-04-01T00:00:00.000000000",
                            "2000-05-01T00:00:00.000000000",
                        ],
                        [
                            "2000-05-01T00:00:00.000000000",
                            "2000-06-01T00:00:00.000000000",
                        ],
                        [
                            "2000-06-01T00:00:00.000000000",
                            "2000-07-01T00:00:00.000000000",
                        ],
                        [
                            "2000-07-01T00:00:00.000000000",
                            "2000-08-01T00:00:00.000000000",
                        ],
                        [
                            "2000-08-01T00:00:00.000000000",
                            "2000-09-01T00:00:00.000000000",
                        ],
                        [
                            "2000-09-01T00:00:00.000000000",
                            "2000-10-01T00:00:00.000000000",
                        ],
                        [
                            "2000-10-01T00:00:00.000000000",
                            "2000-11-01T00:00:00.000000000",
                        ],
                        [
                            "2000-11-01T00:00:00.000000000",
                            "2000-12-01T00:00:00.000000000",
                        ],
                        [
                            "2000-12-01T00:00:00.000000000",
                            "2001-01-01T00:00:00.000000000",
                        ],
                        [
                            "2001-01-01T00:00:00.000000000",
                            "2001-02-01T00:00:00.000000000",
                        ],
                        [
                            "2001-02-01T00:00:00.000000000",
                            "2001-03-01T00:00:00.000000000",
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
                coords={"time": ds.time},
                dims=["time", "bnds"],
                attrs={
                    "xcdat_bounds": "True",
                },
            )

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "DJF"}
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="season",
                data=np.array(
                    [
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                    ],
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )
            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.32608696,
                    0.33695652,
                    0.33695652,
                    0.32967033,
                    0.34065934,
                    0.32967033,
                    0.34444444,
                    0.34444444,
                    0.31111111,
                ]
            )

            assert np.allclose(result, expected, equal_nan=True)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_seasonal_climatology_with_JFD(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "season"
            ds.temporal._weighted = "True"
            ds.temporal._season_config = {"dec_mode": "JDF"}
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="season",
                data=np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                    ],
                ),
                coords={"time": ds.time},
                dims=["time"],
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    [
                        0.17127072,
                        0.16022099,
                        0.33695652,
                        0.32608696,
                        0.33695652,
                        0.32608696,
                        0.33695652,
                        0.33695652,
                        0.32967033,
                        0.34065934,
                        0.32967033,
                        0.17127072,
                        0.17127072,
                        0.15469613,
                        0.17127072,
                    ]
                ]
            )
            assert np.allclose(result, expected, equal_nan=True)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_annual_climatology(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "month"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="month",
                data=np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 3, 1),
                        cftime.DatetimeGregorian(1, 4, 1),
                        cftime.DatetimeGregorian(1, 5, 1),
                        cftime.DatetimeGregorian(1, 6, 1),
                        cftime.DatetimeGregorian(1, 7, 1),
                        cftime.DatetimeGregorian(1, 8, 1),
                        cftime.DatetimeGregorian(1, 9, 1),
                        cftime.DatetimeGregorian(1, 10, 1),
                        cftime.DatetimeGregorian(1, 11, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                        cftime.DatetimeGregorian(1, 1, 1),
                        cftime.DatetimeGregorian(1, 2, 1),
                        cftime.DatetimeGregorian(1, 12, 1),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    [
                        0.5,
                        0.50877193,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.5,
                        0.5,
                        0.49122807,
                        0.5,
                    ]
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)

        def test_weights_for_daily_climatology(self):
            ds = self.ds.copy()

            # Set object attrs required to test the method.
            ds.temporal.dim = "time"
            ds.temporal._mode = "climatology"
            ds.temporal._freq = "day"
            ds.temporal._weighted = "True"
            ds.temporal._time_bounds = ds.time_bnds
            ds.temporal._labeled_time = xr.DataArray(
                name="month_day",
                data=np.array(
                    [
                        cftime.DatetimeGregorian(1, 1, 16),
                        cftime.DatetimeGregorian(1, 2, 15),
                        cftime.DatetimeGregorian(1, 3, 16),
                        cftime.DatetimeGregorian(1, 4, 16),
                        cftime.DatetimeGregorian(1, 5, 6),
                        cftime.DatetimeGregorian(1, 6, 16),
                        cftime.DatetimeGregorian(1, 7, 16),
                        cftime.DatetimeGregorian(1, 8, 16),
                        cftime.DatetimeGregorian(1, 9, 16),
                        cftime.DatetimeGregorian(1, 10, 16),
                        cftime.DatetimeGregorian(1, 11, 16),
                        cftime.DatetimeGregorian(1, 12, 16),
                        cftime.DatetimeGregorian(1, 1, 16),
                        cftime.DatetimeGregorian(1, 2, 15),
                        cftime.DatetimeGregorian(1, 12, 16),
                    ],
                ),
                coords={"time": ds.time},
                attrs={
                    "axis": "T",
                    "long_name": "time",
                    "standard_name": "time",
                    "bounds": "time_bnds",
                },
            )

            # Compare result of the method against the expected.
            result = ds.temporal._get_weights(ds.time_bnds)
            expected = np.array(
                [
                    0.5,
                    0.50877193,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.5,
                    0.49122807,
                    0.5,
                ]
            )
            assert np.allclose(result, expected)

            _check_each_weight_group_adds_up_to_1(ds, result)


class Test_Averager:
    # NOTE: This private method is tested because it is more redundant to
    # test these cases for the public methods that call this private method.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_with_incorrect_mode_arg(self):
        with pytest.raises(ValueError):
            self.ds.temporal._averager(
                "ts",
                "unsupported",
                freq="season",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_freq_arg_is_not_supported_by_operation(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "group_average",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "climatology",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )
        with pytest.raises(ValueError):
            ds.temporal._averager(
                "ts",
                "departures",
                freq="unsupported",
                weighted=True,
                season_config={
                    "dec_mode": "DJF",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )

    def test_raises_error_if_season_config_key_is_not_supported(self):
        with pytest.raises(KeyError):
            self.ds.temporal._averager(
                "ts",
                "climatology",
                freq="season",
                weighted=True,
                season_config={
                    "not_supported": "invalid",
                },
            )

    def test_raises_error_if_december_mode_is_not_supported(self):
        with pytest.raises(ValueError):
            self.ds.temporal._averager(
                "ts",
                "climatology",
                freq="season",
                weighted=True,
                season_config={
                    "dec_mode": "unsupported",
                    "drop_incomplete_djf": False,
                    "custom_seasons": None,
                },
            )


class TestContainsDatetimeLikeObjects:
    def test_returns_false_dataarray_contains_no_datetime_like_objects(self):
        time = xr.DataArray(
            data=np.array([1.0], dtype="float64"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert not _contains_datetime_like_objects(time)

    def test_returns_true_if_dataarray_contains_np_datetime64(self):
        time = xr.DataArray(
            data=np.array(["2000-01-01T12:00:00.000000000"], dtype="datetime64[ns]"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _contains_datetime_like_objects(time)

    def test_returns_true_if_dataarray_contains_np_timedelta64(self):
        time = xr.DataArray(
            data=np.array([86400000000000], dtype="timedelta64[ns]"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _contains_datetime_like_objects(time)

    def test_returns_true_if_dataarray_contains_cftime_datetime(self):
        time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(2000, 1, 1),
                ],
                dtype="object",
            ),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _contains_datetime_like_objects(time)


class TestGetDatetimeLikeType:
    def test_raises_error_if_dataarray_contains_no_datatime_like_objects(self):
        time = xr.DataArray(
            data=np.array([1.0], dtype="float64"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        with pytest.raises(TypeError):
            assert _get_datetime_like_type(time)

    def test_returns_np_datetime64(self):
        time = xr.DataArray(
            data=np.array(["2000-01-01T12:00:00.000000000"], dtype="datetime64[ns]"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _get_datetime_like_type(time) == np.datetime64

    def test_returns_np_timedelta64(self):
        time = xr.DataArray(
            data=np.array([86400000000000], dtype="timedelta64[ns]"),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _get_datetime_like_type(time) == np.timedelta64

    def test_returns_cftime_datetime(self):
        time = xr.DataArray(
            data=np.array(
                [
                    cftime.datetime(2000, 1, 1),
                ],
                dtype="object",
            ),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _get_datetime_like_type(time) == cftime.datetime

        time = xr.DataArray(
            data=np.array(
                [
                    cftime.DatetimeGregorian(2000, 1, 1),
                ],
                dtype="object",
            ),
            dims=["time"],
            attrs={"calendar": "standard", "units": "days since 1850-01-01"},
        )

        assert _get_datetime_like_type(time) == cftime.datetime
