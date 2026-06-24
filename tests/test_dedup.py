import pytest
import numpy as np
import networkx as nx
from binarize2pcalcium.dedup import it_count, del_highest_connected_nodes_without_corr, del_highest_connected_nodes, del_lowest_snr

def test_it_count():
    it = iter([1, 2, 3])
    count, new_it = it_count(it)
    assert count == 3
    assert list(new_it) == [1, 2, 3]

def test_del_highest_connected_nodes_without_corr():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4)])
    connected, removed = del_highest_connected_nodes_without_corr(G)
    assert len(removed) > 0 # At least some nodes removed

def test_del_highest_connected_nodes():
    class DummyC:
        def __init__(self):
            self.corr_threshold = 0.5
            self.G = nx.Graph()
            self.G.add_edges_from([(0, 1), (0, 2), (1, 2)])
            self.F_filtered = np.random.randn(3, 100)
            self.verbose = True
            self.corr_array = np.zeros((3, 3, 2))
            self.corr_array[:, :, 0] = np.array([[1, 0.8, 0.8], [0.8, 1, 0.6], [0.8, 0.6, 1]])

    c = DummyC()
    nn = {0, 1, 2}
    ids, removed = del_highest_connected_nodes(nn, c)
    assert len(removed) > 0

def test_del_highest_connected_nodes_no_tie():
    class DummyC:
        def __init__(self):
            self.corr_threshold = 0.5
            self.G = nx.Graph()
            self.G.add_edges_from([(0, 1), (0, 2)])
            self.F_filtered = np.random.randn(3, 100)
            self.verbose = False
            self.corr_array = np.zeros((3, 3, 2))
            self.corr_array[:, :, 0] = np.array([[1, 0.8, 0.8], [0.8, 1, 0.6], [0.8, 0.6, 1]])

    c = DummyC()
    nn = {0, 1, 2}
    ids, removed = del_highest_connected_nodes(nn, c)
    assert 0 in removed

def test_del_lowest_snr():
    class DummyC:
        def __init__(self):
            self.corr_threshold = 0.5
            self.F_filtered = np.random.randn(3, 100)
            # Make cell 0 have very low signaltonoise
            # signaltonoise is mean / std
            # so we want mean very low, std high
            self.F_filtered[0] = np.random.randn(100) * 10
            self.F_filtered[0] = self.F_filtered[0] - np.mean(self.F_filtered[0])
            self.F_filtered[1] = np.random.randn(100) * 0.1 + 100
            self.F_filtered[2] = np.random.randn(100) * 0.1 + 100
            self.corr_array = np.zeros((3, 3, 2))
            self.corr_array[:, :, 0] = np.array([[1, 0.8, 0.8], [0.8, 1, 0.6], [0.8, 0.6, 1]])

    c = DummyC()
    nn = {0, 1, 2}
    ids, removed = del_lowest_snr(nn, c)
    assert 0 in removed
