"""Top-level package for xcdat."""

from xcdat import tutorial  # noqa: F401
from xcdat.axis import (  # noqa: F401
    center_times,
    get_dim_coords,
    get_dim_keys,
    swap_lon_axis,
)
from xcdat.bounds import BoundsAccessor, create_bounds  # noqa: F401
from xcdat.dataset import decode_time, open_dataset, open_mfdataset  # noqa: F401
from xcdat.regridder.accessor import RegridderAccessor  # noqa: F401
from xcdat.regridder.grid import (  # noqa: F401
    create_axis,
    create_gaussian_grid,
    create_global_mean_grid,
    create_grid,
    create_uniform_grid,
    create_zonal_grid,
)
from xcdat.spatial import SpatialAccessor  # noqa: F401
from xcdat.temporal import TemporalAccessor  # noqa: F401
from xcdat.utils import compare_datasets  # noqa: F401

__version__ = "0.7.3"
