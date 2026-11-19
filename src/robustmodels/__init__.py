"""robustmodels: robust statistics for Python."""

from importlib.metadata import PackageNotFoundError, version

from . import (
    cluster,
    covariance,
    depth,
    glm,
    location,
    multivariate,
    outliers,
    plots,
    rank,
    regression,
    scale,
)

try:
    __version__ = version("robustmodels")
except PackageNotFoundError:  # pragma: no cover (source tree that is not installed)
    __version__ = "unknown"

__all__ = [
    "cluster",
    "covariance",
    "depth",
    "glm",
    "location",
    "multivariate",
    "outliers",
    "plots",
    "rank",
    "regression",
    "scale",
]
