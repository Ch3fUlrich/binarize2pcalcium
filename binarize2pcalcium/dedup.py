"""Cell deduplication functions for calcium imaging analysis.

Functions for finding and removing duplicate/overlapping cells based on
correlation and connection graph analysis.
"""

import itertools
import numpy as np
import networkx as nx
from .correlation import get_correlations
from .thresholds import signaltonoise


def it_count(it):
    """Count elements in an iterator without consuming subsequent elements.

    Args:
        it: Any iterator.

    Returns:
        Tuple of (count, new_iterator_with_same_elements).
    """
    tmp_it, new_it = itertools.tee(it)
    return sum(1 for _ in tmp_it), new_it


def del_highest_connected_nodes_without_corr(G: nx.Graph) -> tuple:
    """Remove highest-ID hub nodes from connected components, ignoring correlation.

    Iteratively removes the most-connected (highest-ID on ties) node from
    each connected component until all edges are resolved.

    Args:
        G: NetworkX graph with cells as nodes.

    Returns:
        Tuple of (connected_cell_ids_list, removed_ids_list).
    """
    connected_components = nx.connected_components(G)
    removed_ids = []
    connected_cell_ids = []
    try:
        while True:
            component = next(connected_components)
            component_edges = G.subgraph(component).edges()
            component_list = list(component_edges)

            while len(component_list) > 0:
                temp = [item for sublist in component_list for item in sublist]
                most_common_elements, counts = np.unique(temp, return_counts=True)
                max_count = np.max(counts)
                max_count_elements = most_common_elements[counts == max_count]
                if len(max_count_elements) > 1:
                    common_element = np.max(max_count_elements)
                else:
                    common_element = max_count_elements[0]

                removed_ids.append(common_element)
                cons_ids = []
                for k in range(len(component_list)):
                    if common_element in component_list[k]:
                        temp_cons = component_list[k]
                        for p in temp_cons:
                            if p != common_element:
                                cons_ids.append(p)
                connected_cell_ids.append(cons_ids)
                component_list = [
                    x for x in component_list if common_element not in x
                ]
    except StopIteration:
        pass

    return connected_cell_ids, removed_ids


def del_highest_connected_nodes(nn: set, c) -> tuple:
    """Remove hub nodes from a correlated component.

    Iteratively removes the node with the most connections (ties broken by
    lowest SNR) until within-group correlation drops below threshold.

    Args:
        nn: Set of cell IDs in the component.
        c: Object with attributes corr_threshold, G, F_filtered, verbose.

    Returns:
        Tuple of (good_cell_ids, removed_cell_ids).
    """
    ids = np.array(list(nn))
    corrs = get_correlations(ids, c.corr_array)
    removed_cells = []

    while np.max(corrs) > c.corr_threshold:
        n_connections = []
        snrs = []
        for n in ids:
            temp1 = signaltonoise(c.F_filtered[n])
            snrs.append(temp1)
            temp2 = c.G.edges([n])
            n_connections.append(len(temp2))

        max_edges = np.max(n_connections)
        idx = np.where(n_connections == max_edges)[0]

        if idx.shape[0] == 1:
            idx2 = np.argmax(n_connections)
            removed_cells.append(ids[idx2])
            ids = np.delete(ids, idx2, 0)
        else:
            snrs = np.array(snrs)
            snrs_idx = snrs[idx]
            idx3 = np.argmin(snrs_idx)
            if c.verbose:
                print("multiple matches found: ", snrs, snrs_idx, idx3)
            removed_cells.append(ids[idx[idx3]])
            ids = np.delete(ids, idx[idx3], 0)

        if ids.shape[0] == 1:
            break

        corrs = get_correlations(ids, c.corr_array)
        if c.verbose:
            print("ids: ", ids, "  corrs: ", corrs)

    return ids, removed_cells


def del_lowest_snr(nn: set, c) -> tuple:
    """Remove lowest-SNR cells from a correlated component.

    Args:
        nn: Set of cell IDs in the component.
        c: Object with attributes corr_threshold, F_filtered.

    Returns:
        Tuple of (good_cell_ids, removed_cell_ids).
    """
    ids = np.array(list(nn))
    corrs = get_correlations(ids, c.corr_array)
    removed_cells = []

    while np.max(corrs) > c.corr_threshold:
        snrs = []
        for n in ids:
            temp = signaltonoise(c.F_filtered[n])
            snrs.append(temp)

        idx = np.argmin(snrs)
        removed_cells.append(ids[idx])
        ids = np.delete(ids, idx, 0)

        if ids.shape[0] == 1:
            break

        corrs = get_correlations(ids, c.corr_array)

    return ids, removed_cells
