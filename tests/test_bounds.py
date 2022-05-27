import cftime
import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset, lat_bnds, lon_bnds
from xcdat.bounds import BoundsAccessor


class TestBoundsAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test__init__(self):
        obj = BoundsAccessor(self.ds)
        assert obj._dataset.identical(self.ds)

    def test_decorator_call(self):
        assert self.ds.bounds._dataset.identical(self.ds)

    def test_map_property_returns_map_of_axis_and_coordinate_keys_to_bounds_dataarray(
        self,
    ):
        ds = self.ds_with_bnds.copy()
        expected = {
            "T": ds.time_bnds,
            "X": ds.lon_bnds,
            "Y": ds.lat_bnds,
            "lat": ds.lat_bnds,
            "latitude": ds.lat_bnds,
            "lon": ds.lon_bnds,
            "longitude": ds.lon_bnds,
            "time": ds.time_bnds,
        }

        result = ds.bounds.map

        for key in expected.keys():
            assert result[key].identical(expected[key])

    def test_keys_property_returns_a_list_of_sorted_bounds_keys(self):
        result = self.ds_with_bnds.bounds.keys
        expected = ["lat_bnds", "lon_bnds", "time_bnds"]

        assert result == expected


class TestAddMissingBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_adds_bounds_in_dataset(self):
        ds = self.ds_with_bnds.copy()

        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        result = ds.bounds.add_missing_bounds()
        assert result.identical(self.ds_with_bnds)

    def test_does_not_fill_bounds_for_coord_of_len_less_than_2(
        self,
    ):
        ds = self.ds_with_bnds.copy()
        ds = ds.isel(time=slice(0, 1))
        ds = ds.drop_vars("time_bnds")

        result = ds.bounds.add_missing_bounds()
        expected = ds.copy()
        assert result.identical(expected)


class TestGetBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_raises_error_when_bounds_dont_exist(self):
        with pytest.raises(KeyError):
            self.ds.bounds.get_bounds("lat")

    def test_getting_existing_bounds_in_dataset(self):
        ds = self.ds_with_bnds.copy()
        lat_bnds = ds.bounds.get_bounds("lat")
        assert lat_bnds.identical(ds.lat_bnds)

        lon_bnds = ds.bounds.get_bounds("lon")
        assert lon_bnds.identical(ds.lon_bnds)
        assert lon_bnds.is_generated

    def test_get_nonexistent_bounds_in_dataset(self):
        ds = self.ds_with_bnds.copy()

        with pytest.raises(KeyError):
            ds = ds.drop_vars(["lat_bnds"])
            ds.bounds.get_bounds("lat")

    def test_raises_error_with_incorrect_coord_arg(self):
        with pytest.raises(ValueError):
            self.ds.bounds.get_bounds("incorrect_coord_argument")


class TestAddBounds:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)
        self.ds_with_bnds = generate_dataset(cf_compliant=True, has_bounds=True)

    def test_add_bounds_raises_error_if_bounds_exist(self):
        ds = self.ds_with_bnds.copy()

        with pytest.raises(ValueError):
            ds.bounds.add_bounds("lat")

    def test_add_bounds_raises_errors_for_data_dim_and_length(self):
        # Multidimensional
        lat = xr.DataArray(
            data=np.array([[0, 1, 2], [3, 4, 5]]),
            dims=["placeholder_1", "placeholder_2"],
            attrs={"units": "degrees_north", "axis": "Y"},
        )
        # Length <=1
        lon = xr.DataArray(
            data=np.array([0]),
            dims=["lon"],
            attrs={"units": "degrees_east", "axis": "X"},
        )
        ds = xr.Dataset(coords={"lat": lat, "lon": lon})

        # If coords dimensions does not equal 1.
        with pytest.raises(ValueError):
            ds.bounds.add_bounds("lat")
        # If coords are length of <=1.
        with pytest.raises(ValueError):
            ds.bounds.add_bounds("lon")

    def test_add_bounds_for_dataset_with_coords_as_datetime_objects(self):
        ds = self.ds.copy()

        result = ds.bounds.add_bounds("lat")
        assert result.lat_bnds.equals(lat_bnds)
        assert result.lat_bnds.is_generated == "True"

        result = result.bounds.add_bounds("lon")
        assert result.lon_bnds.equals(lon_bnds)
        assert result.lon_bnds.is_generated == "True"

        result = ds.bounds.add_bounds("time")
        # NOTE: The algorithm for generating time bounds doesn't extend the
        # upper bound into the next month.
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    ["2000-01-01T12:00:00.000000000", "2000-01-31T12:00:00.000000000"],
                    ["2000-01-31T12:00:00.000000000", "2000-03-01T12:00:00.000000000"],
                    ["2000-03-01T12:00:00.000000000", "2000-03-31T18:00:00.000000000"],
                    ["2000-03-31T18:00:00.000000000", "2000-05-01T06:00:00.000000000"],
                    ["2000-05-01T06:00:00.000000000", "2000-05-31T18:00:00.000000000"],
                    ["2000-05-31T18:00:00.000000000", "2000-07-01T06:00:00.000000000"],
                    ["2000-07-01T06:00:00.000000000", "2000-08-01T00:00:00.000000000"],
                    ["2000-08-01T00:00:00.000000000", "2000-08-31T18:00:00.000000000"],
                    ["2000-08-31T18:00:00.000000000", "2000-10-01T06:00:00.000000000"],
                    ["2000-10-01T06:00:00.000000000", "2000-10-31T18:00:00.000000000"],
                    ["2000-10-31T18:00:00.000000000", "2000-12-01T06:00:00.000000000"],
                    ["2000-12-01T06:00:00.000000000", "2001-01-01T00:00:00.000000000"],
                    ["2001-01-01T00:00:00.000000000", "2001-01-31T06:00:00.000000000"],
                    ["2001-01-31T06:00:00.000000000", "2001-07-17T06:00:00.000000000"],
                    ["2001-07-17T06:00:00.000000000", "2002-05-17T18:00:00.000000000"],
                ],
                dtype="datetime64[ns]",
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)

    def test_returns_bounds_for_dataset_with_coords_as_cftime_objects(self):
        ds = self.ds.copy()
        ds = ds.drop_dims("time")
        ds["time"] = xr.DataArray(
            name="time",
            data=np.array(
                [
                    cftime.DatetimeNoLeap(1850, 1, 1),
                    cftime.DatetimeNoLeap(1850, 2, 1),
                    cftime.DatetimeNoLeap(1850, 3, 1),
                ],
            ),
            dims=["time"],
            attrs={
                "axis": "T",
                "long_name": "time",
                "standard_name": "time",
            },
        )

        result = ds.bounds.add_bounds("time")
        expected_time_bnds = xr.DataArray(
            name="time_bnds",
            data=np.array(
                [
                    [
                        cftime.DatetimeNoLeap(1849, 12, 16, 12),
                        cftime.DatetimeNoLeap(1850, 1, 16, 12),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 1, 16, 12),
                        cftime.DatetimeNoLeap(1850, 2, 15, 0),
                    ],
                    [
                        cftime.DatetimeNoLeap(1850, 2, 15, 0),
                        cftime.DatetimeNoLeap(1850, 3, 15, 0),
                    ],
                ],
            ),
            coords={"time": ds.time.assign_attrs({"bounds": "time_bnds"})},
            dims=["time", "bnds"],
            attrs={"is_generated": "True"},
        )

        assert result.time_bnds.identical(expected_time_bnds)


class Test_GetCoord:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(cf_compliant=True, has_bounds=False)

    def test_gets_coords(self):
        ds = self.ds.copy()

        # Check lat axis coordinates exist
        lat = ds.bounds._get_coords("lat")
        assert lat is not None

        # Check lon axis coordinates exist
        lon = ds.bounds._get_coords("lon")
        assert lon is not None

    def test_raises_error_if_coord_does_not_exist(self):
        ds = self.ds.copy()

        ds = ds.drop_dims("lat")
        with pytest.raises(KeyError):
            ds.bounds._get_coords("lat")
