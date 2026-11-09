import numpy as np
import pytest

from robustmodels import scale


@pytest.mark.parametrize("direct_limit", [0, 50_000])
@pytest.mark.parametrize("tied", [False, True])
def test_every_rank_in_unequal_rows(monkeypatch, direct_limit, tied):
    monkeypatch.setattr(scale, "_DIRECT_LIMIT", direct_limit)
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(7, 19))
    if tied:
        matrix = np.round(matrix)
    matrix.sort(axis=1)
    first = np.array([0, 2, 4, 1, 3, 5, 0])
    last = np.array([18, 17, 16, 18, 15, 4, 18])  # one empty row
    original_first, original_last = first.copy(), last.copy()
    expected = np.sort(
        np.concatenate([row[a : b + 1] for row, a, b in zip(matrix, first, last, strict=True)])
    )
    for rank, value in enumerate(expected, start=1):
        found = scale._kth_smallest(lambda i, j: matrix[i, j], first, last, rank)
        assert found == value
    np.testing.assert_array_equal(first, original_first)
    np.testing.assert_array_equal(last, original_last)


@pytest.mark.parametrize("n", [2, 3, 8, 17])
@pytest.mark.parametrize("diagonal", [False, True])
def test_matrix_median_matches_explicit_pairwise_means(n, diagonal):
    x = np.sort(np.random.default_rng(n).integers(-4, 5, size=n))
    i, j = np.triu_indices(n, k=0 if diagonal else 1)
    expected = np.median((x[i] + x[j]) / 2)
    first, last = scale._pairwise_rows(n, diagonal)
    actual = scale._matrix_median(lambda i, j: (x[i] + x[j]) / 2, first, last)
    assert actual == expected
