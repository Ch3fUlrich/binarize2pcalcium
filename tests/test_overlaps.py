"""Tests for footprint, overlap, and spatial analysis functions."""
import numpy as np
import pytest
from binarize2pcalcium.binarize2pcalcium import (
    Calcium,
    find_overlaps,
    find_overlaps1,
    find_overlaps2,
    make_overlap_database,
    find_inter_cell_distance,
    array_row_intersection,
)


class TestFindOverlaps:
    """Tests for footprint overlap detection functions."""

    def test_find_overlaps_no_overlap(self):
        """Test find_overlaps with non-overlapping footprints."""
        n_cells = 3
        footprints = np.zeros((n_cells, 10, 10))
        footprints[0, 0, 0] = 1
        footprints[0, 0, 1] = 1
        footprints[1, 5, 5] = 1
        footprints[1, 5, 6] = 1
        footprints[2, 9, 9] = 1

        ids = np.arange(n_cells)
        result = find_overlaps(ids, footprints)
        assert len(result) == 0

    def test_find_overlaps_with_overlap(self):
        """Test find_overlaps with overlapping footprints."""
        n_cells = 3
        footprints = np.zeros((n_cells, 10, 10))
        footprints[0, 2, 3] = 1
        footprints[0, 2, 4] = 1
        footprints[1, 2, 3] = 1
        footprints[1, 2, 5] = 1
        footprints[2, 8, 8] = 1

        ids = np.arange(n_cells)
        result = find_overlaps(ids, footprints)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 1

    def test_find_overlaps1(self):
        """Test find_overlaps1 which returns percentage data."""
        n_cells = 3
        footprints = np.zeros((n_cells, 10, 10))
        footprints[0, 2, 3] = 1
        footprints[0, 2, 4] = 1
        footprints[1, 2, 3] = 1

        ids = np.arange(2)
        result = find_overlaps1(ids, footprints)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 1
        assert result[0][2] == 1  # One pixel overlap
        assert result[0][3] == 0.5  # 1 out of 2 pixels

    def test_find_overlaps2(self):
        """Test find_overlaps2 with binarized footprint check."""
        n_cells = 3
        footprints = np.zeros((n_cells, 10, 10))
        footprints_bin = np.zeros((n_cells, 10, 10))
        footprints[0, 3, 4] = 1
        footprints_bin[0, 3, 4] = 1
        footprints[1, 3, 4] = 1
        footprints_bin[1, 3, 4] = 1

        ids = np.arange(2)
        result = find_overlaps2(ids, footprints, footprints_bin)
        assert len(result) == 1


class TestOverlapDatabase:
    """Tests for make_overlap_database."""

    def test_make_overlap_database(self):
        """Test converting overlap results to DataFrame."""
        res = [
            [[0, 1, 10, 0.5, 0.3]],
            [[2, 3, 5, 0.2, 0.1]],
        ]
        df = make_overlap_database(res)
        assert len(df) == 2
        assert list(df.columns) == ['cell1', 'cell2', 'pixels_overlap', 'percent_cell1', 'percent_cell2']
        assert df.iloc[0]['cell1'] == 0
        assert df.iloc[1]['cell1'] == 2

    def test_make_overlap_database_empty(self):
        """Test with empty results."""
        df = make_overlap_database([])
        assert len(df) == 0
        assert set(df.columns) == {'cell1', 'cell2', 'pixels_overlap', 'percent_cell1', 'percent_cell2'}


class TestInterCellDistance:
    """Tests for find_inter_cell_distance."""

    def test_find_inter_cell_distance(self):
        """Test computing distances between cell centres."""
        n_cells = 3
        footprints = np.zeros((n_cells, 10, 10))
        footprints[0, 0, 0] = 1
        footprints[0, 0, 1] = 1
        footprints[1, 0, 5] = 1
        footprints[1, 0, 6] = 1
        footprints[2, 5, 5] = 1

        dists, dists_upper = find_inter_cell_distance(footprints)
        assert dists.shape == (n_cells, n_cells)
        assert dists_upper.shape == (n_cells, n_cells)
        assert dists[0, 1] > 0

    def test_find_inter_cell_distance_self(self):
        """Test self-distance is zero or NaN."""
        n_cells = 2
        footprints = np.zeros((n_cells, 5, 5))
        footprints[0, 1, 1] = 1
        footprints[1, 3, 3] = 1

        dists, _ = find_inter_cell_distance(footprints)
        assert np.isnan(dists[0, 0])
        assert np.isnan(dists[1, 1])
