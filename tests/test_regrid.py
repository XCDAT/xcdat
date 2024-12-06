import datetime
import re
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests import fixtures
from xcdat.regridder import accessor, base, grid, regrid2, xesmf, xgcm

np.set_printoptions(threshold=sys.maxsize, suppress=True)


def gen_uniform_axis(start, stop, step, name, axis):
    temp = np.arange(start, stop, step)

    bounds = np.zeros((temp.shape[0] - 1, 2))
    bounds[:, 0] = temp[:-1]
    bounds[:, 1] = temp[1:]

    points = np.array(
        [temp[i] + ((temp[i + 1] - temp[i]) / 2.0) for i in range(temp.shape[0] - 1)]
    )

    data = xr.DataArray(
        points, dims=[name], attrs={"bounds": f"{name}_bnds", "axis": axis}
    )

    return xr.DataArray(bounds, dims=[name, "bnds"], coords={name: data})


class TestXGCMRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_lev_dataset()

        z = grid.create_axis("lev", np.linspace(10000, 2000, 2), generate_bounds=False)
        self.output_grid = grid.create_grid(z=z)

    def test_multiple_z_axes(self):
        self.ds = self.ds.assign_coords({"ilev": self.ds.lev.copy().rename("ilev")})

        self.ds = self.ds.assign(so_1=self.ds.so.copy().rename("so_1"))

        self.ds["so_1"] = self.ds.so_1.swap_dims({"lev": "ilev"})

        regridder = xgcm.XGCMRegridder(self.ds, self.output_grid, method="linear")

        with pytest.raises(RuntimeError, match=r".*ilev, lev.*"):
            regridder.vertical("so", self.ds)

    def test_vertical_regrid_level_name_mismatch(self):
        self.ds = self.ds.rename({"lev": "plev"})

        regridder = xgcm.XGCMRegridder(self.ds, self.output_grid, method="linear")

        output_data = regridder.vertical("so", self.ds)

        assert output_data["so"].dims == ("time", "lev", "lat", "lon")

    def test_vertical_regrid(self):
        regridder = xgcm.XGCMRegridder(self.ds, self.output_grid, method="linear")

        output_data = regridder.vertical("so", self.ds)

        assert output_data.so.shape == (15, 2, 4, 4)

    @mock.patch("xcdat.regridder.xgcm.Grid")
    def test_target_data(self, grid):
        regridder = xgcm.XGCMRegridder(self.ds, self.output_grid, method="linear")

        regridder.vertical("so", self.ds)

        assert grid.return_value.transform.call_args[0][1] == "Z"

        call_kwargs = grid.return_value.transform.call_args[1]

        assert "method" in call_kwargs and call_kwargs["method"] == "linear"
        assert "target_data" in call_kwargs and call_kwargs["target_data"] is None

    @mock.patch("xcdat.regridder.xgcm.Grid")
    def test_target_data_da(self, grid):
        target_data = np.random.normal(size=self.ds["so"].shape)

        target_da = xr.DataArray(
            target_data, dims=self.ds.so.dims, coords=self.ds.so.coords
        )

        regridder = xgcm.XGCMRegridder(
            self.ds, self.output_grid, method="linear", target_data=target_da
        )

        regridder.vertical("so", self.ds)

        assert grid.return_value.transform.call_args[0][1] == "Z"

        call_kwargs = grid.return_value.transform.call_args[1]

        assert "method" in call_kwargs and call_kwargs["method"] == "linear"
        assert "target_data" in call_kwargs

        xr.testing.assert_allclose(call_kwargs["target_data"], target_da)

    @mock.patch("xcdat.regridder.xgcm.Grid")
    def test_target_data_ds(self, grid):
        target_data = np.random.normal(size=self.ds["so"].shape)

        self.ds["pressure"] = xr.DataArray(
            target_data, dims=self.ds.so.dims, coords=self.ds.so.coords
        )

        regridder = xgcm.XGCMRegridder(
            self.ds, self.output_grid, method="linear", target_data="pressure"
        )

        regridder.vertical("so", self.ds)

        assert grid.return_value.transform.call_args[0][1] == "Z"

        call_kwargs = grid.return_value.transform.call_args[1]

        assert "method" in call_kwargs and call_kwargs["method"] == "linear"
        assert "target_data" in call_kwargs

        xr.testing.assert_allclose(call_kwargs["target_data"], self.ds["pressure"])

    def test_target_data_error(self):
        regridder = xgcm.XGCMRegridder(
            self.ds, self.output_grid, method="linear", target_data="pressure"
        )

        with pytest.raises(
            RuntimeError, match="Could not find target variable 'pressure' in dataset"
        ):
            regridder.vertical("so", self.ds)

    def test_conservative(self):
        regridder = xgcm.XGCMRegridder(
            self.ds, self.output_grid, method="conservative", target_data=None
        )

        with pytest.raises(
            RuntimeError,
            match="Conservative regridding requires a second point position, pass these manually",
        ):
            regridder.vertical("so", self.ds)

    @pytest.mark.parametrize("position", ("left", "center", "right"))
    def test_grid_positions(self, position):
        ds = fixtures.generate_lev_dataset(position=position)

        regridder = xgcm.XGCMRegridder(
            ds,
            self.output_grid,
            method="linear",
            target_data=None,
        )

        output_data = regridder.vertical("so", ds)

        assert output_data.so.shape == (15, 2, 4, 4)

    def test_grid_positions_malformed(self):
        ds = fixtures.generate_lev_dataset(position="malformed")

        regridder = xgcm.XGCMRegridder(
            ds,
            self.output_grid,
            method="linear",
            target_data=None,
        )

        with pytest.raises(
            RuntimeError,
            match="Could not determine the grid point positions, pass these manually",
        ):
            regridder.vertical("so", ds)

    def test_manual_grid_positions(self):
        regridder = xgcm.XGCMRegridder(
            self.ds,
            self.output_grid,
            method="linear",
            target_data=None,
            grid_positions={"left": "lev"},
        )

        output_data = regridder.vertical("so", self.ds)

        assert output_data.so.shape == (15, 2, 4, 4)

    def test_horizontal_placeholder(self):
        regridder = xgcm.XGCMRegridder(
            self.ds, self.output_grid, method="linear", target_data=None
        )

        with pytest.raises(NotImplementedError):
            regridder.horizontal("so", self.ds)

    def test_methods(self):
        xgcm.XGCMRegridder(self.ds, self.output_grid, method="linear", target_data=None)

        with pytest.raises(ValueError, match="'dummy' is invalid, possible choices"):
            xgcm.XGCMRegridder(
                self.ds,
                self.output_grid,
                method="dummy",  # type: ignore
                target_data=None,
            )

    def test_missing_input_z_coord(self):
        ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        regridder = xgcm.XGCMRegridder(
            ds, self.output_grid, method="linear", target_data=None
        )

        with pytest.raises(
            RuntimeError, match="Could not determine `Z` coordinate in dataset."
        ):
            regridder.vertical("ts", ds)

    def test_missing_output_z_coord(self):
        ds = fixtures.generate_lev_dataset()

        self.output_grid = self.output_grid.drop_vars(["lev"])

        regridder = xgcm.XGCMRegridder(
            ds, self.output_grid, method="linear", target_data=None
        )

        with pytest.raises(
            RuntimeError, match="Could not determine 'Z' coordinate in output dataset"
        ):
            regridder.vertical("so", ds)

    def test_missing_input_z_bounds(self):
        ds = fixtures.generate_lev_dataset()

        ds = ds.drop_vars(["lev_bnds"])

        regridder = xgcm.XGCMRegridder(
            ds, self.output_grid, method="linear", target_data=None
        )

        with pytest.raises(
            RuntimeError, match="Could not determine `Z` bounds in dataset."
        ):
            regridder.vertical("so", ds)


class TestRegrid2Regridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.coarse_lat_bnds = gen_uniform_axis(-90, 90.1, 60, "lat", "Y")

        self.fine_lat_bnds = gen_uniform_axis(-90, 90.1, 45, "lat", "Y")

        self.reversed_lat_bnds = gen_uniform_axis(90, -90.1, -60, "lat", "Y")

        self.coarse_lon_bnds = gen_uniform_axis(-0.5, 360, 180, "lon", "X")

        self.fine_lon_bnds = gen_uniform_axis(-0.5, 360, 90, "lon", "X")

        time = pd.date_range("1970-01-01", periods=3)
        self.time_bnds = np.vstack((time[:-1].to_numpy(), time[1:].to_numpy())).reshape(
            (2, 2)
        )
        time = time[:-1].to_pydatetime() + datetime.timedelta(hours=12)

        self.ds_attrs = {
            "Conventions": "CF-1.7 CMIP-6.2",
            "activity_id": "CMIP",
            "experiment": "historical",
        }

        self.da_attrs = {
            "standard_name": "surface_temperature",
            "long_name": "Surface Temperature",
            "units": "K",
        }

        self.coarse_2d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["lat", "lon"],
                    coords={
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                    attrs=self.da_attrs,
                ),
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            },
            attrs=self.ds_attrs,
        )

        self.coarse_3d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            2,
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["time", "lat", "lon"],
                    coords={
                        "time": ("time", time, {"bounds": "time_bnds", "axis": "T"}),
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                    attrs=self.da_attrs,
                ),
                "time_bnds": (["time", "bnds"], self.time_bnds),
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            },
            attrs=self.ds_attrs,
        )

        self.height_bnds = gen_uniform_axis(0.0, 40000.1, 20000.0, "height", "Z")

        self.coarse_4d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones(
                        (
                            2,
                            2,
                            self.coarse_lat_bnds.shape[0],
                            self.coarse_lon_bnds.shape[0],
                        )
                    ),
                    dims=["time", "height", "lat", "lon"],
                    coords={
                        "time": ("time", time, {"bounds": "time_bnds", "axis": "T"}),
                        "height": self.height_bnds["height"],
                        "lat": self.coarse_lat_bnds["lat"],
                        "lon": self.coarse_lon_bnds["lon"],
                    },
                    attrs=self.da_attrs,
                ),
                "time_bnds": (["time", "bnds"], self.time_bnds),
                "height_bnds": self.height_bnds,
                "lat_bnds": self.coarse_lat_bnds,
                "lon_bnds": self.coarse_lon_bnds,
            },
            attrs=self.ds_attrs,
        )

        self.fine_2d_ds = xr.Dataset(
            {
                "ts": xr.DataArray(
                    np.ones((self.fine_lat_bnds.shape[0], self.fine_lon_bnds.shape[0])),
                    dims=["lat", "lon"],
                    coords={
                        "lat": self.fine_lat_bnds["lat"],
                        "lon": self.fine_lon_bnds["lon"],
                    },
                ),
                "lat_bnds": self.fine_lat_bnds,
                "lon_bnds": self.fine_lon_bnds,
            }
        )

    # source ordering is time, height, lat, lon
    @pytest.mark.parametrize(
        "ordering",
        [
            ("time", "lat", "height", "lon"),
            ("lat", "lon", "height", "time"),
            ("lat", "height", "time", "lon"),
            # no change in ordering
            ("time", "height", "lat", "lon"),
        ],
    )
    def test_ordering(self, ordering):
        ds = self.coarse_4d_ds

        output_grid = grid.create_gaussian_grid(4)

        ds["ts"] = ds.ts.transpose(*ordering)

        regridder = regrid2.Regrid2Regridder(ds, output_grid)

        output_ds = regridder.horizontal("ts", ds)

        assert ds.ts.dims == output_ds.ts.dims

    def test_vertical_placeholder(self):
        ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        output_grid = grid.create_gaussian_grid(32)

        regridder = regrid2.Regrid2Regridder(ds, output_grid)

        with pytest.raises(NotImplementedError, match=""):
            regridder.vertical("so", ds)

    @pytest.mark.filterwarnings("ignore:.*invalid value.*divide.*:RuntimeWarning")
    def test_output_bounds(self):
        ds = self.coarse_3d_ds

        output_grid = grid.create_gaussian_grid(32)

        regridder = regrid2.Regrid2Regridder(ds, output_grid)

        output_ds = regridder.horizontal("ts", ds)

        assert "lat_bnds" in output_ds
        assert "lon_bnds" in output_ds
        assert "time_bnds" in output_ds

    @pytest.mark.filterwarnings("ignore:.*invalid value.*divide.*:RuntimeWarning")
    def test_output_bounds_missing_temporal(self):
        ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        ds = self.coarse_3d_ds.drop("time_bnds")

        output_grid = grid.create_gaussian_grid(32)

        regridder = regrid2.Regrid2Regridder(ds, output_grid)

        output_ds = regridder.horizontal("ts", ds)

        assert "lat_bnds" in output_ds
        assert "lon_bnds" in output_ds
        assert "time_bnds" not in output_ds

    @pytest.mark.parametrize(
        "src,dst,expected_west,expected_east,expected_shift",
        [
            (
                np.arange(-180, 180.1, 180),
                np.arange(-180, 180.1, 90),
                np.array([-180, 0, 180]),
                np.array([0, 180, 360]),
                0,
            ),
            (
                np.arange(-180, 180.1, 180),
                np.arange(0, 360.1, 90),
                np.array([0, 180, 360]),
                np.array([180, 360, 540]),
                1,
            ),
            (
                np.arange(0, 360.1, 180),
                np.arange(-180, 180.1, 90),
                np.array([-360, -180, 0]),
                np.array([-180, 0, 180]),
                0,
            ),
            (
                np.arange(0, 360.1, 180),
                np.arange(0, 360.1, 90),
                np.array([0, 180, 360]),
                np.array([180, 360, 540]),
                0,
            ),
            (
                np.arange(180.0, -180.1, -180),
                np.arange(-180, 180.1, 90),
                np.array([0, -180, -360]),
                np.array([-180, -360, -540]),
                1,
            ),
            (
                np.arange(-180.0, 180.1, 180),
                np.arange(180, -180.1, -90),
                np.array([-180, 0, 180]),
                np.array([0, 180, 360]),
                0,
            ),
            (
                np.arange(-360, 360.1, 90),
                np.arange(360, -540.1, -180),
                np.array([-360, -270, -180, -90, -360, -270, -180, -90, 0]),
                np.array([-270, -180, -90, 0, -270, -180, -90, 0, 90]),
                0,
            ),
        ],
    )
    def test_align_axis(self, src, dst, expected_west, expected_east, expected_shift):
        src_west, src_east = src[:-1], src[1:]
        dst_west = dst[:-1]

        shifted_west, shifted_east, shift = regrid2._align_axis(
            src_west, src_east, dst_west
        )

        assert np.all(shifted_west == expected_west)
        assert np.all(shifted_east == expected_east)
        assert shift == expected_shift

    def test_unknown_variable(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        with pytest.raises(KeyError):
            regridder.horizontal("unknown", self.coarse_2d_ds)

    @pytest.mark.filterwarnings("ignore:.*invalid value.*true_divide.*:RuntimeWarning")
    def test_regrid_input_mask(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        self.coarse_2d_ds["mask"] = (("lat", "lon"), [[0, 0], [1, 1], [0, 0]])

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        # np.nan != np.nan, replace with 1e20
        output_data = output_data.fillna(1e20)

        expected_output = np.array(
            [
                [1e20] * 4,
                [1.0] * 4,
                [1.0] * 4,
                [1e20] * 4,
            ],
            dtype=np.float32,
        )

        assert np.all(output_data.ts.values == expected_output)

    @pytest.mark.filterwarnings("ignore:.*invalid value.*true_divide.*:RuntimeWarning")
    def test_regrid_input_mask_unmapped_to_nan(self):
        regridder = regrid2.Regrid2Regridder(
            self.coarse_2d_ds, self.fine_2d_ds, unmapped_to_nan=False
        )

        self.coarse_2d_ds["mask"] = (("lat", "lon"), [[0, 0], [1, 1], [0, 0]])

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        expected_output = np.array(
            [
                [0.0] * 4,
                [1.0] * 4,
                [1.0] * 4,
                [0.0] * 4,
            ],
            dtype=np.float32,
        )

        assert np.all(output_data.ts.values == expected_output)

    def test_preserve_attrs(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        assert output_data.attrs == self.ds_attrs
        assert output_data["ts"].attrs == self.da_attrs

        for x in output_data.coords:
            assert output_data[x].attrs == self.coarse_2d_ds[x].attrs, f"{x}"

    def test_regrid_2d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_2d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_2d_ds)

        assert np.all(output_data.ts == 1)

    def test_regrid_fine_coarse_2d(self):
        regridder = regrid2.Regrid2Regridder(self.fine_2d_ds, self.coarse_2d_ds)

        output_data = regridder.horizontal("ts", self.fine_2d_ds)

        assert np.all(output_data.ts == 1)

    def test_regrid_3d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_3d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_3d_ds)

        assert np.all(output_data.ts == 1)
        assert "lat_bnds" in output_data
        assert "lon_bnds" in output_data
        assert "time_bnds" in output_data

    def test_regrid_4d(self):
        regridder = regrid2.Regrid2Regridder(self.coarse_4d_ds, self.fine_2d_ds)

        output_data = regridder.horizontal("ts", self.coarse_4d_ds)

        assert np.all(output_data.ts == 1)

    def test_map_longitude_coarse_to_fine(self):
        mapping, weights = regrid2._map_longitude(
            self.coarse_lon_bnds.values, self.fine_lon_bnds.values
        )

        expected_mapping = [
            [0],
            [0],
            [1],
            [1],
        ]

        expected_weigths = [
            [[90]],
            [[90]],
            [[90]],
            [[90]],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_longitude_fine_to_coarse(self):
        mapping, weights = regrid2._map_longitude(
            self.fine_lon_bnds.values, self.coarse_lon_bnds.values
        )

        expected_mapping = [
            [0, 1],
            [2, 3],
        ]

        expected_weigths = [[[90, 90]], [[90, 90]]]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_map_latitude_coarse_to_fine(self):
        mapping, weights = regrid2._map_latitude(
            self.coarse_lat_bnds.values, self.fine_lat_bnds.values
        )

        expected_mapping = [
            [
                0,
            ],
            [0, 1],
            [1, 2],
            [
                2,
            ],
        ]

        expected_weigths = [
            [[0.29289322]],
            [[0.20710678], [0.5]],
            [[0.5], [0.20710678]],
            [[0.29289322]],
        ]

        for x, y in zip(mapping, expected_mapping):
            np.testing.assert_allclose(x, y)

        for x2, y2 in zip(weights, expected_weigths):
            np.testing.assert_allclose(x2, y2)

    def test_map_latitude_fine_to_coarse(self):
        mapping, weights = regrid2._map_latitude(
            self.fine_lat_bnds.values, self.coarse_lat_bnds.values
        )

        expected_mapping = [
            [0, 1],
            [1, 2],
            [2, 3],
        ]

        expected_weigths = [
            [[0.29289322], [0.20710678]],
            [[0.5], [0.5]],
            [[0.20710678], [0.29289322]],
        ]

        np.testing.assert_allclose(mapping, expected_mapping)
        np.testing.assert_allclose(weights, expected_weigths)

    def test_extract_bounds(self):
        south, north = regrid2._extract_bounds(self.coarse_lat_bnds)

        assert south.shape == (3,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (3,)
        assert north[0], north[-1] == (60, 90)

    def test_reversed_extract_bounds(self):
        south, north = regrid2._extract_bounds(self.reversed_lat_bnds)

        assert south.shape == (3,)
        assert south[0], south[-1] == (-90, 60)

        assert north.shape == (3,)
        assert north[0], north[-1] == (60, 90)

    def test_get_bounds_ensure_dtype(self):
        del self.coarse_2d_ds.lon.attrs["bounds"]

        with pytest.raises(RuntimeError):
            regrid2._get_bounds_ensure_dtype(self.coarse_2d_ds, "X")


class TestXESMFRegridder:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )
        self.new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

    def test_vertical_placeholder(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        with pytest.raises(NotImplementedError, match=""):
            regridder.vertical("ts", ds)

    def test_regrid(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        output = regridder.horizontal("ts", ds)

        assert output.ts.shape == (15, 46, 73)
        assert "lat_bnds" in output
        assert "lon_bnds" in output
        assert "time_bnds" in output

    @pytest.mark.parametrize(
        "name,value,_",
        [
            ("periodic", True, "_periodic"),
            ("extrap_method", "inverse_dist", "_extrap_method"),
            ("extrap_method", "nearest_s2d", "_extrap_method"),
            ("extrap_dist_exponent", 0.1, "_extrap_dist_exponent"),
            ("extrap_num_src_pnts", 10, "_extrap_num_src_pnts"),
            ("ignore_degenerate", False, "_ignore_degenerate"),
        ],
    )
    def test_flags(self, name, value, _):
        ds = self.ds.copy()

        options = {name: value}

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear", **options)

        assert name in regridder._extra_options
        assert regridder._extra_options[name] == value

    def test_no_variable(self):
        ds = self.ds.copy()

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, "bilinear")

        with pytest.raises(KeyError):
            regridder.horizontal("unknown", ds)

    def test_invalid_method(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(ds, self.new_grid, "bad value")

    def test_invalid_extra_method(self):
        ds = self.ds.copy()

        with pytest.raises(ValueError):
            xesmf.XESMFRegridder(
                ds, self.new_grid, "bilinear", extrap_method="bad value"
            )

    def test_preserve_bounds(self):
        ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=False, has_bounds=True
        )

        ds = ds.drop_vars(["lat_bnds", "lon_bnds"])

        regridder = xesmf.XESMFRegridder(ds, self.new_grid, method="bilinear")

        output = regridder.horizontal("ts", ds)

        assert "time_bnds" in output


class TestGrid:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.lat_data = np.array([-45, 0, 45])
        self.lat = xr.DataArray(self.lat_data.copy(), dims=["lat"], name="lat")

        self.lat_bnds_data = np.array([[-67.5, -22.5], [-22.5, 22.5], [22.5, 67.5]])
        self.lat_bnds = xr.DataArray(
            self.lat_bnds_data.copy(), dims=["lat", "bnds"], name="lat_bnds"
        )

        self.lon_data = np.array([30, 60, 90, 120, 150])
        self.lon = xr.DataArray(self.lon_data.copy(), dims=["lon"], name="lon")

        self.lon_bnds_data = np.array(
            [[15, 45], [45, 75], [75, 105], [105, 135], [135, 165]]
        )
        self.lon_bnds = xr.DataArray(
            self.lon_bnds_data.copy(), dims=["lon", "bnds"], name="lon_bnds"
        )

    def test_create_axis(self):
        expected_axis_attrs = {
            "axis": "Y",
            "units": "degrees_north",
            "coordinate": "latitude",
            "bounds": "lat_bnds",
        }

        axis, bnds = grid.create_axis("lat", self.lat_data)

        assert np.array_equal(axis, self.lat_data)
        assert bnds is not None
        assert bnds.attrs["xcdat_bounds"] == "True"
        assert axis.attrs == expected_axis_attrs

    def test_create_axis_user_attrs(self):
        expected_axis_attrs = {
            "axis": "Y",
            "units": "degrees_south",
            "coordinate": "latitude",
            "bounds": "lat_bnds",
            "custom": "value",
        }

        axis, bnds = grid.create_axis(
            "lat", self.lat_data, attrs={"custom": "value", "units": "degrees_south"}
        )

        assert np.array_equal(axis, self.lat_data)
        assert bnds is not None
        assert bnds.attrs["xcdat_bounds"] == "True"
        assert axis.attrs == expected_axis_attrs

    def test_create_axis_from_list(self):
        axis, bnds = grid.create_axis("lat", self.lat_data, bounds=self.lat_bnds_data)

        assert np.array_equal(axis, self.lat_data)
        assert bnds is not None
        assert np.array_equal(bnds, self.lat_bnds_data)

    def test_create_axis_no_bnds(self):
        expected_axis_attrs = {
            "axis": "Y",
            "units": "degrees_north",
            "coordinate": "latitude",
        }

        axis, bnds = grid.create_axis("lat", self.lat_data, generate_bounds=False)

        assert np.array_equal(axis, self.lat_data)
        assert bnds is None
        assert axis.attrs == expected_axis_attrs

    def test_create_axis_user_bnds(self):
        expected_axis_attrs = {
            "axis": "Y",
            "units": "degrees_north",
            "coordinate": "latitude",
            "bounds": "lat_bnds",
        }

        axis, bnds = grid.create_axis("lat", self.lat_data, bounds=self.lat_bnds_data)

        assert np.array_equal(axis, self.lat_data)
        assert bnds is not None
        assert np.array_equal(bnds, self.lat_bnds_data)
        assert "xcdat_bounds" not in bnds.attrs
        assert axis.attrs == expected_axis_attrs

    def test_create_axis_invalid_name(self):
        with pytest.raises(
            ValueError, match="The name 'mass' is not valid for an axis name."
        ):
            grid.create_axis("mass", self.lat_data)

    def test_empty_grid(self):
        with pytest.raises(
            ValueError, match="Must pass at least 1 axis to create a grid."
        ):
            grid.create_grid()

    def test_create_grid_with_coords(self):
        new_grid = grid.create_grid(x=self.lon, y=self.lat)

        assert np.array_equal(new_grid.lat, self.lat)
        assert np.array_equal(new_grid.lon, self.lon)

    def test_create_grid_with_tuple_of_coords_and_no_bounds(self):
        # This case happens if `create_axis` is used without creating bounds,
        # which will return a tuple of (xr.DataArray, None).
        new_grid = grid.create_grid(x=(self.lon, None), y=(self.lat, None))

        assert np.array_equal(new_grid.lat, self.lat)
        assert np.array_equal(new_grid.lon, self.lon)

    def test_create_grid_with_tuple_of_coords_and_bounds(self):
        new_grid = grid.create_grid(
            x=(self.lon, self.lon_bnds), y=(self.lat, self.lat_bnds)
        )

        assert np.array_equal(new_grid.lat, self.lat)
        assert new_grid.lat.attrs["bounds"] == self.lat_bnds.name
        assert np.array_equal(new_grid.lat_bnds, self.lat_bnds)

        assert np.array_equal(new_grid.lon, self.lon)
        assert new_grid.lon.attrs["bounds"] == self.lon_bnds.name
        assert np.array_equal(new_grid.lon_bnds, self.lon_bnds)

    def test_create_grid_user_attrs(self):
        lev = xr.DataArray(np.linspace(1000, 1, 2), dims=["lev"], name="lev")

        new_grid = grid.create_grid(z=lev, attrs={"custom": "value"})

        assert "custom" in new_grid.attrs
        assert new_grid.attrs["custom"] == "value"

    def test_create_grid_wrong_axis_value(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Argument 'x' should be an xr.DataArray representing coordinates or a tuple (xr.DataArray, xr.DataArray) representing coordinates and bounds."
            ),
        ):
            grid.create_grid(x=(self.lon, self.lon_bnds, self.lat))  # type: ignore[arg-type]

    def test_uniform_grid(self):
        new_grid = grid.create_uniform_grid(-90, 90, 4.0, -180, 180, 5.0)

        assert new_grid.lat[0] == -90.0
        assert new_grid.lat[-1] == 90.0
        assert new_grid.lat.shape == (46,)

        assert new_grid.lon[0] == -180
        assert new_grid.lon[-1] == 180
        assert new_grid.lon.shape == (73,)

    def test_gaussian_grid(self):
        small_grid = grid.create_gaussian_grid(32)

        assert small_grid.lat.shape == (32,)
        assert small_grid.lon.shape == (65,)

        large_grid = grid.create_gaussian_grid(128)

        assert large_grid.lat.shape == (128,)
        assert large_grid.lon.shape == (257,)

        uneven_grid = grid.create_gaussian_grid(33)

        assert uneven_grid.lat.shape == (33,)
        assert uneven_grid.lon.shape == (67,)

    def test_global_mean_grid(self):
        x = grid.create_axis(
            "lon", np.array([0, 45, 90, 180, 270, 360]), generate_bounds=True
        )
        y = grid.create_axis(
            "lat", np.array([-80, -40, 0, 40, 80]), generate_bounds=True
        )

        source_grid = grid.create_grid(x=x, y=y)

        mean_grid = grid.create_global_mean_grid(source_grid)

        assert np.all(mean_grid.lat == np.array([0.0]))
        assert np.all(mean_grid.lat_bnds == np.array([[-90, 90]]))
        assert np.all(mean_grid.lon == np.array([180.0]))
        assert np.all(mean_grid.lon_bnds == np.array([[-22.5, 405]]))

    def test_raises_error_for_global_mean_grid_if_an_axis_has_multiple_dimensions(self):
        source_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims="lat",
                    attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([0, 45, 90, 180, 270, 360]),
                    dims="lon",
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [
                            [-90.0, -60.0],
                            [-60.0, -20.0],
                            [-20.0, 20.0],
                            [20.0, 60.0],
                            [60.0, 90.0],
                        ]
                    ),
                    dims=["lat", "bnds"],
                    attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [
                            [-22.5, 22.5],
                            [22.5, 67.5],
                            [67.5, 135.0],
                            [135.0, 225.0],
                            [225.0, 315.0],
                            [315.0, 405.0],
                        ]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": True},
                ),
            },
        )

        source_grid_with_2_lats = source_grid.copy()
        source_grid_with_2_lats["lat2"] = xr.DataArray(
            name="lat2",
            data=np.array([-80, -40, 0, 40, 80]),
            dims="lat2",
            attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
        )
        with pytest.raises(ValueError):
            grid.create_global_mean_grid(source_grid_with_2_lats)

        source_grid_with_2_lons = source_grid.copy()
        source_grid_with_2_lons["lon2"] = xr.DataArray(
            name="lon2",
            data=np.array([0, 45, 90, 180, 270, 360]),
            dims="lon2",
            attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
        )
        with pytest.raises(ValueError):
            grid.create_global_mean_grid(source_grid_with_2_lons)

    def test_zonal_grid(self):
        x = grid.create_axis(
            "lon", np.array([-160, -80, 80, 160]), generate_bounds=True
        )
        y = grid.create_axis(
            "lat", np.array([-80, -40, 0, 40, 80]), generate_bounds=True
        )

        source_grid = grid.create_grid(x=x, y=y)

        zonal_grid = grid.create_zonal_grid(source_grid)

        assert np.all(zonal_grid.lat == np.array([-80, -40, 0, 40, 80]))
        assert np.all(
            zonal_grid.lat_bnds
            == np.array([[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]])
        )
        assert np.all(zonal_grid.lon == np.array([0.0]))
        assert np.all(zonal_grid.lon_bnds == np.array([-200, 200]))

    def test_raises_error_for_zonal_grid_if_an_axis_has_multiple_dimensions(self):
        source_grid = xr.Dataset(
            coords={
                "lat": xr.DataArray(
                    name="lat",
                    data=np.array([-80, -40, 0, 40, 80]),
                    dims="lat",
                    attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
                ),
                "lon": xr.DataArray(
                    name="lon",
                    data=np.array([-160, -80, 80, 160]),
                    dims="lon",
                    attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
                ),
            },
            data_vars={
                "lat_bnds": xr.DataArray(
                    name="lat_bnds",
                    data=np.array(
                        [[-90, -60], [-60, -20], [-20, 20], [20, 60], [60, 90]]
                    ),
                    dims=["lat", "bnds"],
                    attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
                ),
                "lon_bnds": xr.DataArray(
                    name="lon_bnds",
                    data=np.array(
                        [[-200.0, -120.0], [-120.0, 0.0], [0.0, 120.0], [120.0, 200.0]]
                    ),
                    dims=["lon", "bnds"],
                    attrs={"xcdat_bounds": True},
                ),
            },
        )

        source_grid_with_2_lats = source_grid.copy()
        source_grid_with_2_lats["lat2"] = xr.DataArray(
            name="lat2",
            data=np.array([-80, -40, 0, 40, 80]),
            dims="lat2",
            attrs={"units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
        )
        with pytest.raises(ValueError):
            grid.create_zonal_grid(source_grid_with_2_lats)

        source_grid_with_2_lons = source_grid.copy()
        source_grid_with_2_lons["lon2"] = xr.DataArray(
            name="lon2",
            data=np.array([0, 45, 90, 180, 270, 360]),
            dims="lon2",
            attrs={"units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
        )
        with pytest.raises(ValueError):
            grid.create_zonal_grid(source_grid_with_2_lons)


class TestAccessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = mock.MagicMock()
        self.ac = accessor.RegridderAccessor(self.data)
        self.horizontal_ds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=True, has_bounds=True
        )
        self.vertical_ds = fixtures.generate_lev_dataset()

    def test_preserve_mask_from_input(self):
        mask = xr.zeros_like(self.horizontal_ds.isel(time=0), dtype=int)
        mask = mask.drop_vars("time")

        mask.ts[1:3] = 1.0

        self.horizontal_ds["mask"] = mask.ts

        grid = accessor._get_input_grid(self.horizontal_ds, "ts", ["X", "Y"])

        assert "mask" in grid

        xr.testing.assert_allclose(mask.ts, grid.mask)

    def test_horizontal(self):
        output_grid = grid.create_gaussian_grid(32)

        output_data = self.horizontal_ds.regridder.horizontal(
            "ts", output_grid, tool="xesmf", method="bilinear"
        )

        assert output_data.ts.shape == (15, 32, 65)

        # use dataset with time dimension
        output_data = self.horizontal_ds.regridder.horizontal(
            "ts", self.horizontal_ds, tool="xesmf", method="bilinear"
        )

        assert output_data.ts.shape == (15, 4, 4)

        ds_multi = fixtures.generate_multiple_variable_dataset(
            1,
            separate_dims=True,
            decode_times=True,
            cf_compliant=True,
            has_bounds=True,
        )

        output_data = ds_multi.regridder.horizontal(
            "ts", self.horizontal_ds, tool="xesmf", method="bilinear"
        )

        assert output_data.ts.shape == (15, 4, 4)

    def test_vertical(self):
        z = grid.create_axis("lev", np.linspace(10000, 2000, 2), generate_bounds=False)

        output_grid = grid.create_grid(z=z)

        output_data = self.vertical_ds.regridder.vertical(
            "so", output_grid, tool="xgcm", method="linear"
        )

        assert output_data.so.shape == (15, 2, 4, 4)

        # use dataset with time dimension
        output_data = self.vertical_ds.regridder.vertical(
            "so", self.vertical_ds, tool="xgcm", method="linear"
        )

        assert output_data.so.shape == (15, 4, 4, 4)

    def test_vertical_multiple_z_axes(self):
        z = grid.create_axis("lev", np.linspace(10000, 2000, 2), generate_bounds=False)
        output_grid = grid.create_grid(z=z)

        self.vertical_ds = self.vertical_ds.assign_coords(
            {"ilev": self.vertical_ds.lev.copy().rename("ilev")}
        )

        self.vertical_ds = self.vertical_ds.assign(
            so_1=self.vertical_ds.so.copy().rename("so_1")
        )

        self.vertical_ds["so_1"] = self.vertical_ds.so_1.swap_dims({"lev": "ilev"})

        output_data = self.vertical_ds.regridder.vertical(
            "so", output_grid, tool="xgcm", method="linear"
        )

        assert output_data.so.shape == (15, 2, 4, 4)

    def test_grid(self):
        ds_bounds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=True, has_bounds=True
        )

        grid = ds_bounds.regridder.grid

        assert "lat" in grid
        assert "lon" in grid
        assert "lat_bnds" in grid
        assert "lon_bnds" in grid

        ds_no_bounds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=True, has_bounds=False
        )

        grid = ds_no_bounds.regridder.grid

        assert "lat" in grid
        assert "lon" in grid
        assert "lat_bnds" in grid
        assert "lon_bnds" in grid

        ds_multi = fixtures.generate_multiple_variable_dataset(
            1, separate_dims=True, decode_times=True, cf_compliant=True, has_bounds=True
        )

        with pytest.raises(
            ValueError,
            match=r".*lon\d?.*lon\d?.*",
        ):
            ds_multi.regridder.grid  # noqa: B018

    def test_grid_raises_error_when_dataset_has_multiple_dims_for_an_axis(self):
        ds_bounds = fixtures.generate_dataset(
            decode_times=True, cf_compliant=True, has_bounds=True
        )
        ds_bounds.coords["lat2"] = xr.DataArray(
            data=[], dims="lat2", attrs={"axis": "Y"}
        )

        with pytest.raises(ValueError):
            ds_bounds.regridder.grid  # noqa: B018

    @mock.patch("xcdat.regridder.accessor._get_input_grid")
    def test_horizontal_tool_check(self, _get_input_grid):
        mock_regridder = mock.MagicMock()
        mock_regridder.return_value.horizontal.return_value = "output data"

        mock_data = mock.MagicMock()

        with mock.patch.dict(
            accessor.HORIZONTAL_REGRID_TOOLS, {"regrid2": mock_regridder}
        ):
            output = self.ac.horizontal("ts", mock_data, tool="regrid2")

        assert output == "output data"

        mock_regridder.return_value.horizontal.assert_called_with("ts", self.data)

        with pytest.raises(
            ValueError, match=r"Tool 'dummy' does not exist, valid choices"
        ):
            self.ac.horizontal("ts", mock_data, tool="dummy")  # type: ignore

    @mock.patch("xcdat.regridder.accessor._get_input_grid")
    def test_vertical_tool_check(self, _get_input_grid):
        mock_regridder = mock.MagicMock()
        mock_regridder.return_value.vertical.return_value = "output data"

        mock_data = mock.MagicMock()

        with mock.patch.dict(accessor.VERTICAL_REGRID_TOOLS, {"xgcm": mock_regridder}):
            output = self.ac.vertical("ts", mock_data, tool="xgcm", target_data=None)

        assert output == "output data"

        mock_regridder.return_value.vertical.assert_called_with("ts", self.data)

        with pytest.raises(
            ValueError, match=r"Tool 'dummy' does not exist, valid choices"
        ):
            self.ac.vertical("ts", mock_data, tool="dummy", target_data=None)  # type: ignore


class TestBase:
    def test_preserve_bounds(self):
        output_grid = fixtures.generate_lev_dataset()

        input_ds = output_grid.copy(deep=True)
        input_ds.lat_bnds.attrs["source"] = "input_ds"
        input_ds.lon_bnds.attrs["source"] = "input_ds"
        input_ds.time_bnds.attrs["source"] = "input_ds"
        input_ds.lev_bnds.attrs["source"] = "input_ds"

        output_grid = output_grid.drop_vars(["time_bnds", "lev_bnds"])
        output_grid.lat_bnds.attrs["source"] = "output_grid"
        output_grid.lon_bnds.attrs["source"] = "output_grid"

        target = xr.Dataset()

        output_ds = base._preserve_bounds(input_ds, output_grid, target, ["X", "Y"])

        assert "lat_bnds" in output_ds
        assert output_ds.lat_bnds.attrs["source"] == "output_grid"
        assert "lon_bnds" in output_ds
        assert output_ds.lon_bnds.attrs["source"] == "output_grid"
        assert "time_bnds" in output_ds
        assert output_ds.time_bnds.attrs["source"] == "input_ds"
        assert "lev_bnds" in output_ds
        assert output_ds.lev_bnds.attrs["source"] == "input_ds"

    def test_preserve_bounds_does_not_drop_axis_if_axis_does_not_exist(self):
        output_grid = fixtures.generate_lev_dataset()
        output_grid = output_grid.drop_dims("lat")

        input_ds = output_grid.copy(deep=True)

        target = xr.Dataset()
        output_ds = base._preserve_bounds(input_ds, output_grid, target, ["Y"])

        # Check that lat is still not in the dimensions (nothing happens).
        assert "lat" not in output_ds.dims

    def test_regridder_implementation(self):
        class NewRegridder(base.BaseRegridder):
            def __init__(self, src_grid, dst_grid, **options):
                super().__init__(src_grid, dst_grid, **options)

            def vertical(self, data_var, ds):
                return ds

            def horizontal(self, data_var, ds):
                return ds

        regridder = NewRegridder(mock.MagicMock(), mock.MagicMock())

        assert regridder is not None

        ds_in = mock.MagicMock()

        ds_out = regridder.horizontal("ts", ds_in)

        assert ds_in == ds_out
