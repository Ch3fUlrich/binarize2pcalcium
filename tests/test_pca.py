import os
import pytest
import numpy as np
from binarize2pcalcium.pca import run_UMAP, compute_PCA, compute_TSNE, compute_UMAP, fit_curves_aucs, fit_curves_general, load_pca_animal

def test_run_UMAP():
    data = np.random.randn(10, 5)
    u = run_UMAP(data, n_neighbors=5, min_dist=0.1, n_components=2)
    assert u.shape == (10, 2)

def test_compute_PCA(tmp_path):
    X = np.random.randn(10, 5)
    d = tmp_path / "pca"
    d.mkdir()

    pca, X_pca = compute_PCA(X, str(d), suffix1="test_", suffix2="1_")
    assert X_pca.shape == (10, 5)
    assert os.path.exists(d / "test_1_pca.pkl")
    assert os.path.exists(d / "test_1_pca.npy")

    # test load
    pca_load, X_pca_load = compute_PCA(X, str(d), suffix1="test_", suffix2="1_", recompute=False)
    assert np.allclose(X_pca, X_pca_load)

def test_compute_PCA_not_save(tmp_path):
    X = np.random.randn(10, 5)
    d = tmp_path / "pca_not_save"
    d.mkdir()

    pca, X_pca = compute_PCA(X, str(d), suffix1="test_", suffix2="1_", save=False)
    assert X_pca.shape == (10, 5)

def test_compute_TSNE(tmp_path):
    X = np.random.randn(10, 5)
    d = tmp_path / "tsne"
    d.mkdir()

    X_tsne = compute_TSNE(X, str(d), n_components=2, perplexity=5)
    assert X_tsne.shape == (10, 2)
    assert os.path.exists(d / "tsne.npz")

    # test load
    X_tsne_load = compute_TSNE(X, str(d), n_components=2, perplexity=5)
    assert np.allclose(X_tsne, X_tsne_load)

def test_compute_UMAP(tmp_path):
    X = np.random.randn(10, 5)
    d = tmp_path / "umap"
    d.mkdir()

    X_umap = compute_UMAP(X, str(d), n_components=2, text="test_")
    assert X_umap.shape == (10, 2)
    assert os.path.exists(d / "test_umap.npz")

    # test load
    X_umap_load = compute_UMAP(X, str(d), n_components=2, text="test_")
    assert np.allclose(X_umap, X_umap_load)

def test_load_pca_animal(tmp_path):
    d = tmp_path / "animal1"
    d.mkdir()

    fname = d / "pca_test_random_True_quiescent_False_aucs.npz"
    np.savez(fname, aucs=[1,2,3], n_neurons=10)

    aucs, n_neurons = load_pca_animal(str(tmp_path), "animal1", "test", True, False)
    assert np.allclose(aucs, [1, 2, 3])
    assert n_neurons == 10

    # Missing file
    aucs, n_neurons = load_pca_animal(str(tmp_path), "animal1", "missing", True, False)
    assert aucs is None

def test_fit_curves_aucs():
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    aucs = [100, 150, 200, 250, np.nan]
    fit_curves_aucs(aucs, fig1, ax1, ax2, fig3, ax3, "animal1", ".", "test_method", ["red"])
    assert ax2.get_title().startswith("Pearson corr")

def test_fit_curves_general():
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        "Mouse_id": ["animal1"],
        "Pday_start": [0],
        "Pday_end": [10],
        "Group": ["test"]
    })
    aucs = [100, 150, 200, 250]
    fig, ax = plt.subplots()

    fit_curves_general(df, aucs, ax, "animal1", "red")
    lines = ax.get_lines()
    assert len(lines) == 1
