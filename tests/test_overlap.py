import pytest
import numpy as np
import pandas as pd
from binarize2pcalcium.overlap import find_overlaps, find_overlaps1, find_overlaps2, make_overlap_database, find_inter_cell_distance, alpha_shape, array_row_intersection

def test_array_row_intersection():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[3, 4], [5, 6]])
    res = array_row_intersection(a, b)
    assert len(res) == 1
    assert np.allclose(res[0], [3, 4])

def test_find_overlaps():
    ids = np.array([0, 1])
    footprints = np.zeros((2, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 2, 3] = 1
    res = find_overlaps(ids, footprints)
    assert len(res) == 1
    assert res[0][0] == 0
    assert res[0][1] == 1
    assert len(res[0][2]) == 1 # overlap pixels

def test_find_overlaps1():
    ids = np.array([0, 1])
    footprints = np.zeros((2, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 2, 3] = 1
    res = find_overlaps1(ids, footprints)
    assert len(res) == 1
    assert res[0][0] == 0
    assert res[0][1] == 1
    assert res[0][2] == 1 # number of overlap pixels
    assert res[0][3] == 1.0 # pct cell 1
    assert res[0][4] == 1.0 # pct cell 2

def test_find_overlaps2():
    ids = np.array([0, 1])
    footprints = np.zeros((2, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 2, 3] = 1
    res = find_overlaps2(ids, footprints, footprints) # Use footprints as bin
    assert len(res) == 1
    assert res[0][0] == 0
    assert res[0][1] == 1

def test_find_overlaps2_no_overlap():
    ids = np.array([0, 1])
    footprints = np.zeros((2, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 4, 5] = 1
    res = find_overlaps2(ids, footprints, footprints) # Use footprints as bin
    assert len(res) == 0

def test_make_overlap_database():
    res = [[[0, 1, 10, 0.5, 0.5], [0, 2, 5, 0.2, 0.1]]]
    df = make_overlap_database(res)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "cell1" in df.columns

def test_find_inter_cell_distance():
    footprints = np.zeros((2, 10, 10))
    footprints[0, 2, 3] = 1
    footprints[1, 2, 5] = 1
    dists, dists_upper = find_inter_cell_distance(footprints)
    assert dists.shape == (2, 2)
    assert np.isnan(dists[0, 0])
    assert dists[0, 1] == 2.0

def test_alpha_shape():
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]])
    geom, edges = alpha_shape(points, alpha=0.1)
    assert len(edges) > 0

    points_small = np.array([[0, 0], [0, 1], [1, 0]])
    geom_small = alpha_shape(points_small)
    assert geom_small is not None
