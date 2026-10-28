"""Original-image size metadata must retain H/W and the case index."""

import numpy as np
import pytest

from src.datasets.batch import original_size_from_metadata


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("sizes", [
    [(101, 203)],
    [(101, 203), (307, 409)],
    [(101, 203), (307, 409), (511, 613)],
])
def test_default_collate_size_vectors(container, sizes):
    heights = np.asarray([h for h, _ in sizes], dtype=np.int64)
    widths = np.asarray([w for _, w in sizes], dtype=np.int64)
    metadata = {"orig_size": container((heights, widths))}
    for idx, expected in enumerate(sizes):
        assert original_size_from_metadata(metadata, idx) == expected


@pytest.mark.parametrize("sizes", [
    [(101, 203)],
    [(101, 203), (307, 409)],
    [(101, 203), (307, 409), (511, 613)],
])
def test_batch_by_two_array_has_cases_on_first_axis(sizes):
    metadata = {"orig_size": np.asarray(sizes, dtype=np.int64)}
    for idx, expected in enumerate(sizes):
        assert original_size_from_metadata(metadata, idx) == expected


@pytest.mark.parametrize("size", [
    (53, 79),
    [53, 79],
    np.asarray([53, 79], dtype=np.int64),
])
def test_single_sample_size(size):
    assert original_size_from_metadata({"orig_size": size}, 0) == (53, 79)


@pytest.mark.parametrize("metadata", [None, {}, {"case_id": "case"}, {"orig_size": None}])
def test_absent_size_returns_none(metadata):
    assert original_size_from_metadata(metadata, 0) is None


def test_resolved_dimensions_are_plain_python_integers():
    actual = original_size_from_metadata({"orig_size": [np.asarray([53]), np.asarray([79])]}, 0)
    assert tuple(type(value) for value in actual) == (int, int)


def test_unbatched_pair_can_describe_a_shared_batch_size():
    assert original_size_from_metadata({"orig_size": (53, 79)}, 4) == (53, 79)


@pytest.mark.parametrize("size", [
    [],
    [53],
    [53, 79, 101],
    np.zeros((2, 3), dtype=np.int64),
    np.zeros((1, 2, 2), dtype=np.int64),
    (0, 79),
    (-1, 79),
    (53.5, 79),
    (float("nan"), 79),
    (float("inf"), 79),
    (True, 79),
])
def test_invalid_size_cannot_silently_resize_a_prediction(size):
    with pytest.raises(ValueError):
        original_size_from_metadata({"orig_size": size}, 0)


@pytest.mark.parametrize("size", [
    np.asarray([[53, 79]], dtype=np.int64),
    [np.asarray([53]), np.asarray([79])],
])
def test_batched_size_index_must_exist(size):
    with pytest.raises(IndexError):
        original_size_from_metadata({"orig_size": size}, 1)
