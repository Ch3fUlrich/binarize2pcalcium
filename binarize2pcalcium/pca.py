"""PCA and dimensionality reduction for calcium imaging data.

Functions for UMAP, TSNE, PCA computation, and multi-session
longitudinal analysis of neural population dynamics.
"""

import os
import numpy as np
from tqdm import tqdm
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats
import matplotlib.pyplot as plt


def run_UMAP(
    data: np.ndarray,
    n_neighbors: int = 50,
    min_dist: float = 0.1,
    n_components: int = 3,
    metric: str = 'euclidean',
) -> np.ndarray:
    """Run UMAP dimensionality reduction.

    Args:
        data: 2D array [n_samples, n_features].
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        n_components: Number of output dimensions.
        metric: Distance metric.

    Returns:
        UMAP-transformed array [n_samples, n_components].
    """
    import umap
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    return u


def compute_PCA(
    X: np.ndarray,
    data_dir: str,
    suffix1: str = '',
    suffix2: str = '',
    recompute: bool = True,
    save: bool = True,
) -> tuple:
    """Run PCA and optionally save results.

    Args:
        X: 2D data array [n_samples, n_features].
        data_dir: Directory to save/load PCA results.
        suffix1: Filename suffix part 1.
        suffix2: Filename suffix part 2.
        recompute: If True, recompute even if file exists.
        save: If True, save results to disk.

    Returns:
        Tuple of (pca_object, X_pca_transformed).
    """
    import pickle as pk
    from sklearn.decomposition import PCA

    fname_out = os.path.join(data_dir, str(suffix1) + str(suffix2) + 'pca.pkl')

    if not os.path.exists(fname_out) or recompute:
        print(f"Running PCA (saving flag: {save}, location: {fname_out})")
        pca = PCA()
        X_pca = pca.fit_transform(X)

        if save:
            pk.dump(pca, open(fname_out, "wb"))
            np.save(fname_out.replace('pkl', 'npy'), X_pca)
        else:
            print("... not saving...")
    else:
        with open(fname_out, 'rb') as file:
            pca = pk.load(file)
        X_pca = np.load(fname_out.replace('pkl', 'npy'))

    return pca, X_pca


def compute_TSNE(
    X: np.ndarray,
    data_dir: str,
    n_components: int = 2,
    perplexity: float = 100,
    learning_rate: float = 10,
) -> np.ndarray:
    """Run TSNE dimensionality reduction with result caching.

    Args:
        X: 2D data array [n_samples, n_features].
        data_dir: Directory for caching results.
        n_components: Output dimensions.
        perplexity: TSNE perplexity.
        learning_rate: TSNE learning rate.

    Returns:
        TSNE-transformed array.
    """
    from sklearn.manifold import TSNE

    fname_out = os.path.join(data_dir, 'tsne.npz')

    try:
        data = np.load(fname_out, allow_pickle=True)
        X_tsne_gpu = data['X_tsne_gpu']
    except FileNotFoundError:
        X_tsne_gpu = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
        ).fit_transform(X)

        np.savez(
            fname_out,
            X_tsne_gpu=X_tsne_gpu,
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
        )

    return X_tsne_gpu


def compute_UMAP(
    X: np.ndarray,
    root_dir: str,
    n_components: int = 3,
    text: str = '',
) -> np.ndarray:
    """Run UMAP with result caching.

    Args:
        X: 2D data array.
        root_dir: Root directory for caching.
        n_components: Output dimensions.
        text: Prefix for cache filename.

    Returns:
        UMAP-transformed array.
    """
    fname_out = os.path.join(root_dir, text + 'umap.npz')

    try:
        data = np.load(fname_out, allow_pickle=True)
        X_umap = data['X_umap']
    except FileNotFoundError:
        n_neighbors = 50
        min_dist = 0.1
        metric = 'euclidean'

        X_umap = run_UMAP(X, n_neighbors, min_dist, n_components, metric)

        np.savez(
            fname_out,
            X_umap=X_umap,
            n_components=n_components,
        )

    return X_umap


def fit_curves_aucs(
    aucs: list,
    fig1, ax1, ax2,
    fig3, ax3,
    animal_id: str,
    root_dir: str,
    binarization_method: str,
    clrs: list,
):
    """Fit linear regression to AUC values for longitudinal analysis.

    Args:
        aucs: List of AUC values per session.
        fig1, ax1, ax2, fig3, ax3: Matplotlib figure/axes handles.
        animal_id: Animal identifier.
        root_dir: Root data directory.
        binarization_method: Method name for labeling.
        clrs: Color list.
    """
    regr_auc = linear_model.LinearRegression()

    aucs = np.array(aucs).squeeze()
    idx = np.where(np.isnan(aucs) == False)[0]
    t = idx
    aucs = aucs[idx]

    pcor = scipy.stats.pearsonr(t, aucs)
    ax2.set_title(
        "Pearson corr: " + str(round(pcor[0], 6)) + " " + str(round(pcor[1], 6))
    )

    regr_auc.fit(t.reshape(-1, 1), aucs.reshape(-1, 1))
    pred_y = regr_auc.predict(t.reshape(-1, 1))

    ax2.plot(t, pred_y, c='black', linewidth=3, label=binarization_method)
    ax2.legend()

    fontsize = 10
    plt.title(
        animal_id + ", " + ", auc slope " + str(round(regr_auc.coef_[0][0], 4)),
        fontsize=fontsize,
    )

    ax1.set_ylim(0, 400)
    ax1.set_xlim(0, 10)
    ax1.set_xlabel("Session Day (chronological)", fontsize=fontsize)
    ax1.set_ylabel("CIRCLES - # of detected cells", fontsize=fontsize)
    ax1.tick_params(axis='both', which='both', labelsize=fontsize)

    ax2.tick_params(axis='both', which='both', labelsize=fontsize)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("TRIANGLES - Area under variance explained curve", fontsize=fontsize)


def fit_curves_general(
    df: "pd.DataFrame",
    aucs: list,
    ax,
    animal_id: str,
    clr,
):
    """Fit linear regression for general AUC-vs-time analysis.

    Args:
        df: DataFrame with Mouse_id, Pday_start, Pday_end, Group columns.
        aucs: List of AUC values.
        ax: Matplotlib axes.
        animal_id: Animal identifier.
        clr: Color.

    Returns:
        Matplotlib axes.
    """
    regr_auc = linear_model.LinearRegression()

    aucs = np.array(aucs)
    idx = np.where(np.isnan(aucs) == False)[0]
    aucs = aucs[idx]
    x = idx.reshape(-1, 1)

    pcor = scipy.stats.pearsonr(np.arange(len(aucs)), np.array(aucs))

    y = np.array(aucs).reshape(-1, 1)
    regr_auc.fit(x, y)
    pred_y = regr_auc.predict(x)

    idx2 = np.where(df['Mouse_id'] == animal_id)[0].squeeze()
    P_start = int(df.iloc[idx2]['Pday_start'])
    P_end = int(df.iloc[idx2]['Pday_end'])
    age = df.iloc[idx2]['Group']

    xx = idx + P_start
    pval = pcor[1]

    if pval < 0.01:
        alpha = 1.0
    elif pval < 0.05:
        alpha = 0.8
    else:
        alpha = 0.4

    line_type = '' if pval < 0.05 else '--'

    ax.scatter(xx, aucs, c=clr, s=100, alpha=1)

    ax.plot(
        xx,
        pred_y,
        line_type,
        c=clr,
        linewidth=3,
        label=animal_id + " " + age + ", Pcor: " +
        str(format(pcor[0], ".3f")) + ", Pval: " + str(format(pcor[1], ".3f")),
        alpha=alpha,
    )

    return ax


def load_pca_animal(
    root_dir: str,
    animal_id: str,
    binarization_method: str,
    cell_randomization: bool,
    quiescent: bool,
) -> tuple:
    """Load PCA AUC results for a specific animal.

    Args:
        root_dir: Root data directory.
        animal_id: Animal identifier.
        binarization_method: Name of binarization method.
        cell_randomization: Whether cell randomization was used.
        quiescent: Whether quiescent-only periods were used.

    Returns:
        Tuple of (aucs, n_neurons) or (None, None) if file not found.
    """
    fnames_aucs = os.path.join(
        root_dir,
        animal_id,
        'pca_' + binarization_method + '_random_' +
        str(cell_randomization) + '_quiescent_' + str(quiescent) + '_aucs.npz',
    )

    if os.path.exists(fnames_aucs):
        data = np.load(fnames_aucs)
        aucs = data['aucs']
        n_neurons = data['n_neurons']
    else:
        print("File does not exist", fnames_aucs)
        return None, None

    return aucs, n_neurons
