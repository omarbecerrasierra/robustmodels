"""robustmodels: robust statistics for Python."""

from importlib.metadata import PackageNotFoundError, version

from . import covariance, location, scale

try:
    __version__ = version("robustmodels")
except PackageNotFoundError:  # pragma: no cover (source tree that is not installed)
    __version__ = "unknown"

__all__ = ["covariance", "location", "scale"]
