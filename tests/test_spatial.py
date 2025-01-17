import numpy as np
import pytest
import xarray as xr

from tests import requires_dask
from tests.fixtures import generate_dataset
from xcdat.spatial import SpatialAccessor


class TestSpatialAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test__init__(self):
        ds = self.ds.copy()
        obj = SpatialAccessor(ds)

        assert obj._dataset.identical(ds)

    def test_decorator_call(self):
        ds = self.ds.copy()
        obj = ds.spatial

        assert obj._dataset.identical(ds)


class TestAverage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        # Limit to just 3 data points to simplify testing.
        self.ds = self.ds.isel(time=slice(None, 3))

        # Change the value of the first element so that it is easier to identify
        # changes in the output.
        self.ds["ts"].data[0] = np.full((4, 4), 2.25)

    def test_raises_error_if_data_var_not_in_dataset(self):
        with pytest.raises(KeyError):
            self.ds.spatial.average("not_a_data_var", axis=["Y", "incorrect_axis"])

    def test_raises_error_if_axis_list_contains_unsupported_axis(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y", "incorrect_axis"])

    def test_raises_error_if_lat_axis_coords_cant_be_found(self):
        ds = self.ds.copy()

        # Update CF metadata to invalid values so cf_xarray can't interpret them.
        del ds.lat.attrs["axis"]
        del ds.lat.attrs["standard_name"]
        del ds.lat.attrs["units"]
        # Update coordinate name.
        ds = ds.rename({"lat": "invalid_lat"})
        ds = ds.set_index(invalid_lat="invalid_lat")

        with pytest.raises(KeyError):
            ds.spatial.average("ts", axis=["X", "Y"])

    def test_raises_error_if_lon_axis_coords_cant_be_found(self):
        ds = self.ds.copy()

        # Update CF metadata to invalid values so cf_xarray can't interpret them.
        del ds.lon.attrs["axis"]
        del ds.lon.attrs["standard_name"]
        del ds.lon.attrs["units"]
        # Update coordinate name.
        ds = ds.rename({"lon": "invalid_lon"})
        ds = ds.set_index(invalid_lon="invalid_lon")

        with pytest.raises(KeyError):
            ds.spatial.average("ts", axis=["X", "Y"])

    def test_raises_error_if_bounds_type_is_not_a_tuple(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=[1, 1])

        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds="str")

    def test_raises_error_if_there_are_0_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=())

    def test_raises_error_if_there_are_more_than_two_elements_in_the_bounds(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(1, 2, 3))

    def test_raises_error_if_lower_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=("invalid", 1))

    def test_raises_error_if_upper_bound_is_not_a_float_or_int(self):
        with pytest.raises(TypeError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(1, "invalid"))

    def test_raises_error_if_lower_lat_bound_is_larger_than_upper(self):
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["Y"], lat_bounds=(2, 1))

    def test_does_not_raise_error_if_lower_lon_bound_is_larger_than_upper(self):
        self.ds.spatial.average("ts", axis=["X"], lon_bounds=(2, 1))

    def test_raises_error_if_lat_axis_is_specified_but_lat_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lon": self.ds.lon}, dims=["lon"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_raises_error_if_lon_axis_is_specified_but_lon_is_not_in_weights_dims(
        self,
    ):
        weights = xr.DataArray(
            data=np.ones(4), coords={"lat": self.ds.lat}, dims=["lat"]
        )
        with pytest.raises(KeyError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_raises_error_if_weights_lat_and_lon_dims_dont_align_with_data_var_dims(
        self,
    ):
        # Get a slice of the dataset to reduce the size of the dimensions for
        # simpler testing.
        ds = self.ds.isel(lat=slice(0, 3), lon=slice(0, 3))
        weights = xr.DataArray(
            data=np.ones((3, 3)),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )

        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["X", "Y"], weights=weights)

    def test_raises_error_if_min_weight_not_between_zero_and_one(
        self,
    ):
        # ensure error if min_weight less than zero
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["X", "Y"], min_weight=-0.01)

        # ensure error if min_weight greater than 1
        with pytest.raises(ValueError):
            self.ds.spatial.average("ts", axis=["X", "Y"], min_weight=1.01)

    def test_spatial_average_for_lat_region_and_keep_weights(self):
        ds = self.ds.copy()

        result = ds.spatial.average(
            "ts", axis=["Y"], lat_bounds=(-5.0, 5), keep_weights=True
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )
        expected["lat_wts"] = xr.DataArray(
            name="lat_wts",
            data=np.array([0.0, 0.08715574, 0.08715574, 0.0]),
            dims=["lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_region(self):
        ds = self.ds.copy()

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.average("ts", axis=["Y"], lat_bounds=(-5.0, 5))

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )

        assert result.identical(expected)

    def test_spatial_average_for_domain_wrapping_p_meridian_non_cf_conventions(
        self,
    ):
        ds = self.ds.copy()

        # get spatial average for original dataset
        ref = ds.spatial.average("ts").ts

        # change first bound from -0.9375 to 359.0625
        lon_bnds = ds.lon_bnds.copy()
        lon_bnds[0, 0] = 359.0625
        ds["lon_bnds"] = lon_bnds

        # check spatial average with new (bad) bound
        result = ds.spatial.average("ts").ts

        assert result.identical(ref)

    @requires_dask
    def test_spatial_average_for_lat_region_and_keep_weights_with_dask(self):
        ds = self.ds.copy().chunk(2)

        # Specifying axis as a str instead of list of str.
        result = ds.spatial.average(
            "ts", axis=["Y"], lat_bounds=(-5.0, 5), keep_weights=True
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array(
                [[2.25, 2.25, 2.25, 2.25], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            coords={"time": expected.time, "lon": expected.lon},
            dims=["time", "lon"],
        )
        expected["lat_wts"] = xr.DataArray(
            name="lat_wts",
            data=np.array([0.0, 0.08715574, 0.08715574, 0.0]),
            dims=["lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_and_lon_region_and_keep_weights(self):
        ds = self.ds.copy()
        result = ds.spatial.average(
            "ts",
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            keep_weights=True,
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )
        expected["lat_lon_wts"] = xr.DataArray(
            name="ts_weights",
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 4.34907156, 4.34907156, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            dims=["lon", "lat"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_with_min_weight(self):
        ds = self.ds.copy()

        # insert a nan
        ds["ts"][0, :, 2] = np.nan

        result = ds.spatial.average(
            "ts",
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            min_weight=1.0,
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([np.nan, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_with_min_weight_as_None(self):
        ds = self.ds.copy()

        result = ds.spatial.average(
            "ts",
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            min_weight=None,
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        xr.testing.assert_allclose(result, expected)

    def test_spatial_average_for_lat_and_lon_region_with_custom_weights(self):
        ds = self.ds.copy()

        weights = xr.DataArray(
            data=np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12], [4, 8, 12, 16]]),
            coords={"lat": ds.lat, "lon": ds.lon},
            dims=["lat", "lon"],
        )
        result = ds.spatial.average(
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            weights=weights,
            data_var="ts",
        )

        expected = self.ds.copy()
        expected["ts"] = xr.DataArray(
            data=np.array([2.25, 1.0, 1.0]),
            coords={"time": expected.time},
            dims="time",
        )

        assert result.identical(expected)


class TestGetWeights:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_value_error_thrown_for_multiple_out_of_order_lon_bounds(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array([[3, 1], [5, 3], [5, 7], [7, 9]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        # Check _get_longitude_weights raises error when there are
        # > 1 out-of-order bounds for the dataset.
        with pytest.raises(ValueError):
            self.ds.spatial._get_longitude_weights(domain_bounds, region_bounds=None)

    def test_raises_error_if_dataset_has_multiple_bounds_variables_for_an_axis(self):
        ds = self.ds.copy()

        # Create a second "Y" axis dimension and associated bounds
        ds["lat2"] = ds.lat.copy()
        ds["lat2"].name = "lat2"
        ds["lat2"].attrs["bounds"] = "lat_bnds2"
        ds["lat_bnds2"] = ds.lat_bnds.copy()
        ds["lat_bnds2"].name = "lat_bnds2"

        # Check raises error when there are > 1 bounds for the dataset.
        with pytest.raises(TypeError):
            ds.spatial.get_weights(axis=["Y", "X"])

    def test_data_var_weights_for_region_in_lat_and_lon_domains(self):
        ds = self.ds.copy()

        result = ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=(-5, 5), lon_bounds=(-170, -120), data_var="ts"
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_dataset_weights_for_region_in_lat_and_lon_domains(self):
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=(-5, 5), lon_bounds=(-170, -120)
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 4.35778714, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_dataset_weights_for_region_in_lat_domain(self):
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=(-5, 5), lon_bounds=None
        )
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.16341702, 15.52461668, 15.52461668, 0.16341702],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_weights_for_region_in_lon_domain(self):
        expected = xr.DataArray(
            data=np.array(
                [
                    [0.0, 0.0, 0.00297475, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 49.99702525, 0.0],
                    [0.0, 0.0, 0.00297475, 0.0],
                ]
            ),
            coords={"lat": self.ds.lat, "lon": self.ds.lon},
            dims=["lat", "lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["Y", "X"], lat_bounds=None, lon_bounds=(-170, -120)
        )

        xr.testing.assert_allclose(result, expected)

    def test_dataset_weights_for_region_in_lon_domain_with_region_spanning_p_meridian(
        self,
    ):
        ds = self.ds.copy()

        result = ds.spatial._get_longitude_weights(
            domain_bounds=ds.lon_bnds,
            # Region spans prime meridian.
            region_bounds=np.array([359, 1]),
        )
        expected = xr.DataArray(
            data=np.array([1.875, 0.0625, 0.0, 0.0625]),
            coords={"lon": ds.lon},
            dims=["lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_dataset_weights_all_longitudes_for_equal_region_bounds(self):
        expected = xr.DataArray(
            data=np.array(
                [1.875, 178.125, 178.125, 1.875],
            ),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["X"], lat_bounds=None, lon_bounds=np.array([0.0, 360.0])
        )

        xr.testing.assert_allclose(result, expected)

    def test_dataset_weights_for_equal_region_bounds_representing_entire_lon_domain(
        self,
    ):
        expected = xr.DataArray(
            data=np.array(
                [1.875, 178.125, 178.125, 1.875],
            ),
            coords={"lon": self.ds.lon},
            dims=["lon"],
        )
        result = self.ds.spatial.get_weights(
            axis=["X"], lat_bounds=None, lon_bounds=np.array([10.0, 10.0])
        )

        xr.testing.assert_allclose(result, expected)


class Test_SwapLonAxis:
    # NOTE: This private method is tested because we might want to consider
    # converting it to a public method in the future.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_raises_error_with_incorrect_orientation_to_swap_to(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )
        with pytest.raises(ValueError):
            self.ds.spatial._swap_lon_axis(domain, to=9000)

    @requires_dask
    def test_swap_chunked_domain_dataarray_from_180_to_360(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=360)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[295, 355], [355, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

    @requires_dask
    def test_swap_chunked_domain_dataarray_from_360_to_180(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, 181], [181, 360]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, -179], [-179, 0]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, 359.75]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        ).chunk(2)

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, -0.25]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

    def test_swap_domain_dataarray_from_180_to_360(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-65, -5], [-5, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=360)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[295, 355], [355, 0], [0, 120]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

    def test_swap_domain_dataarray_from_360_to_180(self):
        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, 181], [181, 360]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[0, 120], [120, -179], [-179, 0]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

        domain = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, 359.75]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        result = self.ds.spatial._swap_lon_axis(domain, to=180)
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array([[-0.25, 120], [120, -0.25]]),
            dims=["lon", "bnds"],
            attrs={"xcdat_bounds": "True"},
        )

        assert result.identical(expected)

    def test_swap_region_ndarray_from_180_to_360(self):
        result = self.ds.spatial._swap_lon_axis(np.array([-65, 0, 120]), to=360)
        expected = np.array([295, 0, 120])

        assert np.array_equal(result, expected)

        result = self.ds.spatial._swap_lon_axis(np.array([-180, 0, 180]), to=360)
        expected = np.array([180, 0, 180])

        assert np.array_equal(result, expected)

    def test_swap_region_ndarray_from_360_to_180(self):
        result = self.ds.spatial._swap_lon_axis(np.array([0, 120, 181, 360]), to=180)
        expected = np.array([0, 120, -179, 0])

        assert np.array_equal(result, expected)

        result = self.ds.spatial._swap_lon_axis(np.array([-0.25, 120, 359.75]), to=180)
        expected = np.array([-0.25, 120, -0.25])

        assert np.array_equal(result, expected)


class Test_ScaleDimToRegion:
    # NOTE: This private method is tested because this is an in-house algorithm
    # that has edge cases with some complexities.
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    @requires_dask
    def test_scales_chunked_lat_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        ).chunk(2)

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([-5, 5])
        )
        expected = xr.DataArray(
            name="lat_bnds",
            data=np.array([[-5.0, -5.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    @requires_dask
    def test_scales_chunked_lon_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        ).chunk(2)

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([190, 240])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [[240.0, 240.0], [190.0, 190.0], [190.0, 240.0], [240.0, 240.0]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )
        assert result.identical(expected)

    def test_scales_lat_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lat_bnds",
            data=np.array(
                [[-90, -89.375], [-89.375, 0.0], [0.0, 89.375], [89.375, 90]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([-5, 5])
        )
        expected = xr.DataArray(
            name="lat_bnds",
            data=np.array([[-5.0, -5.0], [-5.0, 0.0], [0.0, 5.0], [5.0, 5.0]]),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    def test_scales_lon_bounds_when_not_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    [359.0625, 360.9375],
                    [0.9375, 179.0625],
                    [179.0625, 357.1875],
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([190, 240])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [[240.0, 240.0], [190.0, 190.0], [190.0, 240.0], [240.0, 240.0]]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)

    def test_scales_lon_bounds_when_wrapping_around_prime_meridian(self):
        domain_bounds = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 179.0625],
                    # Grid cells in between boundaries.
                    [179.0625, 357.1875],
                    # Grid cell straddling lower boundary.
                    [357.1875, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        result = self.ds.spatial._scale_domain_to_region(
            domain_bounds=domain_bounds, region_bounds=np.array([357.5, 10.0])
        )
        expected = xr.DataArray(
            name="lon_bnds",
            data=np.array(
                [
                    # Does not apply to any conditional.
                    [359.0625, 360.9375],
                    # Grid cells stradling upper boundary.
                    [0.9375, 10.0],
                    # Grid cells in between boundaries.
                    [10.0, 10.0],
                    # Grid cell straddling lower boundary.
                    [357.5, 359.0625],
                ]
            ),
            coords={"lat": self.ds.lat},
            dims=["lat", "bnds"],
        )

        assert result.identical(expected)
