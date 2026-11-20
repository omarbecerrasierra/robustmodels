"""The packaged data sets have the shapes, names and values of their R sources."""

import json
from pathlib import Path

import numpy as np
import pytest

from robustmodels import datasets

DATA = Path(__file__).parent / "data"
REFERENCE = json.loads((DATA / "r_reference.json").read_text())["multivariate"]
GROUPS = json.loads((DATA / "r_multivariate.json").read_text())["groups"]


@pytest.mark.parametrize(
    "loader, key, shape, target",
    [
        (datasets.load_stars, "starsCYG", (47, 1), "log.light"),
        (datasets.load_hbk, "hbk", (75, 3), "Y"),
        (datasets.load_wood, "wood", (20, 5), "y"),
        (datasets.load_stackloss, "stackloss", (21, 3), "stack.loss"),
    ],
)
def test_regression_data_match_the_r_reference(loader, key, shape, target):
    data = loader()
    assert data.X.shape == shape and data.y.shape == (shape[0],)
    assert len(data.feature_names) == shape[1] and data.target_name == target
    reference = np.array(REFERENCE[key]["X"])
    columns = np.column_stack([data.X, data.y])[:, : reference.shape[1]]
    np.testing.assert_array_equal(columns, reference)


def test_hemophilia_has_group_labels():
    X, y, names, target = datasets.load_hemophilia()
    np.testing.assert_array_equal(X, GROUPS["hemophilia"]["X"])
    assert names == ("AHFactivity", "AHFantigen") and target == "gr"
    assert y.dtype.kind == "U"
    assert (y == "normal").sum() == 30 and (y == "carrier").sum() == 45
