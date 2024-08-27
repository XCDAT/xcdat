import numpy as np
import pytest
import xarray as xr

from tests.fixtures import generate_dataset
from xcdat.utils import (
    _validate_min_weight,
    compare_datasets,
    mask_var_with_weight_threshold,
    str_to_bool,
)


class TestCompareDatasets:
    def test_returns_unique_coord_and_data_var_keys(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(name="data_var1"),
            },
            coords={"coord1": xr.DataArray(name="coord1")},
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(name="data_var1"),
                "data_var2": xr.DataArray(name="data_var2"),
            },
            coords={
                "coord1": xr.DataArray(name="coord1"),
                "coord2": xr.DataArray(name="coord2"),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert result["unique_coords"] == ["coord2"]
        assert result["unique_data_vars"] == ["data_var2"]

    def test_returns_nonidentical_and_nonequal_coord_and_data_var_keys(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[1]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[1]),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert sorted(result["nonidentical_coords"]) == ["coord1", "coord2"]
        assert sorted(result["nonequal_coords"]) == ["coord1", "coord2"]
        assert sorted(result["nonidentical_data_vars"]) == ["data_var1", "data_var2"]
        assert sorted(result["nonequal_data_vars"]) == ["data_var1", "data_var2"]

    def test_returns_no_differences_between_datasets(self):
        ds1 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )
        ds2 = xr.Dataset(
            data_vars={
                "data_var1": xr.DataArray(data=[0]),
                "data_var2": xr.DataArray(data=[0]),
            },
            coords={
                "coord1": xr.DataArray(data=[0]),
                "coord2": xr.DataArray(data=[0]),
            },
        )

        result = compare_datasets(ds1, ds2)
        assert result["nonidentical_coords"] == []
        assert result["nonequal_coords"] == []
        assert result["nonidentical_data_vars"] == []
        assert result["nonequal_data_vars"] == []


class TestStrToBool:
    def test_converts_str_to_bool(self):
        result = str_to_bool("True")
        expected = True
        assert result == expected

        result = str_to_bool("False")
        expected = False
        assert result == expected

    def test_raises_error_if_str_is_not_a_python_bool(self):
        with pytest.raises(ValueError):
            str_to_bool(True)  # type: ignore

        with pytest.raises(ValueError):
            str_to_bool(1)  # type: ignore

        with pytest.raises(ValueError):
            str_to_bool("1")


class TestMaskVarWithWeightThreshold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

    def test_returns_mask_var_with_spatial_min_weight_of_100(self):
        ds = self.ds.copy()
        ds = ds.isel({"time": slice(0, 3), "lat": slice(0, 3), "lon": slice(0, 3)})
        ds["ts"][0, :, 2] = np.nan

        # Function arguments.
        dv = ds["ts"].copy()
        weights = ds.spatial.get_weights(
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            data_var="ts",
        )

        result = mask_var_with_weight_threshold(dv, weights, min_weight=1.0)
        expected = xr.DataArray(
            data=np.array(
                [
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                    ],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
            coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
            dims=["time", "lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_returns_mask_var_with_spatial_min_weight_of_0(self):
        ds = self.ds.copy()
        ds = ds.isel({"time": slice(0, 3), "lat": slice(0, 3), "lon": slice(0, 3)})

        # Function arguments.
        dv = ds["ts"].copy()
        weights = ds.spatial.get_weights(
            axis=["X", "Y"],
            lat_bounds=(-5.0, 5),
            lon_bounds=(-170, -120.1),
            data_var="ts",
        )

        result = mask_var_with_weight_threshold(dv, weights, min_weight=0)
        expected = xr.DataArray(
            data=np.array(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
            coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
            dims=["time", "lat", "lon"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_returns_mask_var_with_temporal_min_weight_of_100(self):
        ds = self.ds.copy()
        ds = ds.isel({"time": slice(0, 3), "lat": slice(0, 3), "lon": slice(0, 3)})
        ds["ts"][0, :, 2] = np.nan

        # Function arguments.
        dv = ds["ts"].copy()
        weights = xr.DataArray(
            name="time_wts",
            data=np.array([1.0, 1.0, 1.0]),
            dims="time",
            coords={"time": ds.time},
        )

        result = mask_var_with_weight_threshold(dv, weights, min_weight=0)
        expected = xr.DataArray(
            data=np.array(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]],
                ]
            ),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["lat", "lon", "time"],
        )

        xr.testing.assert_allclose(result, expected)

    def test_returns_mask_var_with_temporal_min_weight_of_0(self):
        ds = self.ds.copy()
        ds = ds.isel({"time": slice(0, 3), "lat": slice(0, 3), "lon": slice(0, 3)})

        # Function arguments.
        dv = ds["ts"].copy()
        weights = xr.DataArray(
            name="time_wts",
            data=np.array([1.0, 1.0, 1.0]),
            dims="time",
            coords={"time": ds.time},
        )

        result = mask_var_with_weight_threshold(dv, weights, min_weight=0)
        expected = xr.DataArray(
            data=np.array(
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ]
            ),
            coords={"lat": ds.lat, "lon": ds.lon, "time": ds.time},
            dims=["lat", "lon", "time"],
        )

        xr.testing.assert_allclose(result, expected)


class TestValidateMinWeight:
    def test_pass_None_returns_0(self):
        result = _validate_min_weight(None)

        assert result == 0

    def test_returns_error_if_less_than_0(self):
        with pytest.raises(ValueError):
            _validate_min_weight(-1)

    def test_returns_error_if_greater_than_1(self):
        with pytest.raises(ValueError):
            _validate_min_weight(1.1)

    def test_returns_valid_min_weight(self):
        result = _validate_min_weight(1)

        assert result == 1
