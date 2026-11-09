import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))  # makes `reference` importable


def _samples():
    """Normal, skewed, tied and constant samples with odd and even n."""
    rng = np.random.default_rng(2026)
    out = []
    for k in range(400):
        n = int(rng.integers(2, 41))
        kind = k % 5
        if kind == 0:
            x = rng.normal(size=n)
        elif kind == 1:
            x = rng.lognormal(size=n)
        elif kind == 2:
            x = rng.integers(0, 6, size=n).astype(float)
        elif kind == 3:
            x = rng.normal(size=n)
            x[: max(2, n // 3)] = np.median(x)
        else:
            x = np.full(n, 2.5)
        out.append(x.round(6))
    return out


@pytest.fixture(scope="session")
def samples():
    return _samples()
