"""Tests for overlap module."""
import numpy as np
import pytest
from binarize2pcalcium.overlap import (
    array_row_intersection,
    find_overlaps,
    find_overlaps1,
    find_overlaps2,
    make_overlap_database,
    find_inter_cell_distance,
)


def test_array_row_intersection():
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[3, 4], [7, 8], [5, 6]])
    result = array_row_intersection(a, b)
    assert result.shape[0] >= 2


def test_array_row_intersection_no_common():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    result = array_row_intersection(a, b)
    assert len(result) == 0


def test_find_overlaps_no_overlap():
    footprints = np.zeros((3, 10, 10))
    footprints[0, 0, 0] = 1
    footprints[1, 5, 5] = 1
    footprints[2, 9, 9] = 1
    result = find_overlaps(np.arange(3), footprints)
    assert len(result) == 0


def test_find_overlaps_with_overlap():
    footprints = np.zeros((3, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 2, 3] = 1
    footprints[2, 8, 8] = 1
    result = find_overlaps(np.arange(3), footprints)
    assert len(result) == 1
    assert result[0][0] == 0
    assert result[0][1] == 1


def test_find_overlaps1():
    footprints = np.zeros((3, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[0, 2, 4] = 1
    footprints[1, 2, 3] = 1
    result = find_overlaps1(np.arange(2), footprints)
    assert len(result) == 1
    assert result[0][2] == 1


def test_make_overlap_database():
    res = [[[0, 1, 10, 0.5, 0.3]], [[2, 3, 5, 0.2, 0.1]]]
    df = make_overlap_database(res)
    assert len(df) == 2
    assert list(df.columns) == ['cell1', 'cell2', 'pixels_overlap', 'percent_cell1', 'percent_cell2']


def test_find_inter_cell_distance():
    footprints = np.zeros((3, 10, 10))
    footprints[0, 0, 0] = 1
    footprints[1, 0, 5] = 1
    footprints[2, 5, 5] = 1
    dists, dists_upper = find_inter_cell_distance(footprints)
    assert dists.shape == (3, 3)
    assert dists[0, 1] > 0
    assert np.isnan(dists[0, 0])
