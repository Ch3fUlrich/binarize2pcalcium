"""Spatial overlap analysis for calcium imaging footprints.

Functions for finding overlapping pixels between cell footprints,
computing inter-cell distances, and building overlap databases.
"""

import numpy as np
import sklearn.metrics


def array_row_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find rows in array `a` that also exist in array `b`.

    Args:
        a: 2D array of coordinates.
        b: 2D array of coordinates.

    Returns:
        2D array of common rows from `a`.
    """
    tmp = np.prod(np.swapaxes(a[:, :, None], 1, 2) == b, axis=2)
    return a[np.sum(np.cumsum(tmp, axis=0) * tmp == 1, axis=1).astype(bool)]


def find_overlaps(ids: np.ndarray, footprints: np.ndarray) -> list:
    """Find overlapping pixels between cell footprints.

    Args:
        ids: Array of cell IDs to check.
        footprints: 3D array [n_cells, height, width] of footprint masks.

    Returns:
        List of [cell1, cell2, overlapping_pixels] entries.
    """
    intersections = []
    for k in ids:
        temp = footprints[k]
        idx1 = np.vstack(np.where(temp > 0)).T

        for p in range(k + 1, footprints.shape[0], 1):
            temp = footprints[p]
            idx2 = np.vstack(np.where(temp > 0)).T

            res = array_row_intersection(idx1, idx2)
            if len(res) > 0:
                intersections.append([k, p, res])

    return intersections


def find_overlaps1(ids: np.ndarray, footprints: np.ndarray) -> list:
    """Find overlapping pixels with percentage metrics.

    Args:
        ids: Array of cell IDs to check.
        footprints: 3D array [n_cells, height, width] of footprint masks.

    Returns:
        List of [cell1, cell2, overlap_pixels, pct_cell1, pct_cell2] entries.
    """
    intersections = []
    for k in ids:
        temp1 = footprints[k]
        idx1 = np.vstack(np.where(temp1 > 0)).T

        for p in range(k + 1, footprints.shape[0], 1):
            temp2 = footprints[p]
            idx2 = np.vstack(np.where(temp2 > 0)).T
            res = array_row_intersection(idx1, idx2)

            if len(res) > 0:
                percent1 = res.shape[0] / idx1.shape[0]
                percent2 = res.shape[0] / idx2.shape[0]
                intersections.append([k, p, res.shape[0], percent1, percent2])

    return intersections


def find_overlaps2(
    ids: np.ndarray, footprints: np.ndarray, footprints_bin: np.ndarray
) -> list:
    """Find overlapping pixels with binarized pre-check.

    Skips cell pairs whose binarized footprints don't overlap.

    Args:
        ids: Array of cell IDs to check.
        footprints: 3D array [n_cells, height, width].
        footprints_bin: 3D binarized footprint array.

    Returns:
        List of [cell1, cell2, overlap_pixels, pct_cell1, pct_cell2] entries.
    """
    intersections = []
    for k in ids:
        temp1 = footprints[k]
        idx1 = np.vstack(np.where(temp1 > 0)).T
        temp1_bin = footprints_bin[k]

        for p in range(k + 1, footprints.shape[0], 1):
            temp2 = footprints[p]
            idx2 = np.vstack(np.where(temp2 > 0)).T
            temp2_bin = footprints_bin[p]

            if np.max(temp1_bin + temp2_bin) < 2:
                continue

            res = array_row_intersection(idx1, idx2)

            if len(res) > 0:
                percent1 = res.shape[0] / idx1.shape[0]
                percent2 = res.shape[0] / idx2.shape[0]
                intersections.append([k, p, res.shape[0], percent1, percent2])

    return intersections


def make_overlap_database(res: list) -> "pd.DataFrame":
    """Convert overlap results list into a pandas DataFrame.

    Args:
        res: List of overlap result lists.

    Returns:
        DataFrame with columns: cell1, cell2, pixels_overlap, percent_cell1, percent_cell2.
    """
    import pandas as pd

    data = []
    for k in range(len(res)):
        for p in range(len(res[k])):
            data.append(res[k][p])

    df = pd.DataFrame(
        data,
        columns=[
            'cell1',
            'cell2',
            'pixels_overlap',
            'percent_cell1',
            'percent_cell2',
        ],
    )
    return df


def find_inter_cell_distance(
    footprints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pairwise Euclidean distances between cell centres.

    Centre is computed as median of non-zero footprint pixels.

    Args:
        footprints: 3D array [n_cells, height, width].

    Returns:
        Tuple of (distance_matrix, upper_triangle_matrix).
        Self-distance entries are set to NaN.
    """
    locations = []
    for k in range(footprints.shape[0]):
        temp = footprints[k]
        centre = np.median(np.vstack(np.where(temp > 0)).T, axis=0)
        locations.append(centre)

    locations = np.vstack(locations)
    dists = sklearn.metrics.pairwise.euclidean_distances(locations)

    dists_upper = np.triu(dists, -1)
    idx = np.where(dists == 0)
    dists[idx] = np.nan

    return dists, dists_upper


def alpha_shape(points: np.ndarray, alpha: float = 0.6):
    """Compute the alpha shape (concave hull) of a set of points.

    Args:
        points: Nx2 array of point coordinates.
        alpha: Alpha value for concavity. Smaller = more detail.

    Returns:
        Tuple of (unary_union_geometry, edge_points_list).
    """
    from shapely.ops import unary_union, polygonize
    from scipy.spatial import Delaunay
    import shapely.geometry as geometry

    if len(points) < 4:
        return geometry.MultiPoint(list(points)).convex_hull

    coords = points
    tri = Delaunay(coords)
    triangles = coords[tri.simplices]
    a = ((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2 +
         (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2) ** 0.5
    b = ((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2 +
         (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2) ** 0.5
    c = ((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2 +
         (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2) ** 0.5
    s = (a + b + c) / 2.0
    areas = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = np.unique(
        np.concatenate((edge1, edge2, edge3)), axis=0
    ).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles_list = list(polygonize(m))

    return unary_union(triangles_list), edge_points
