"""Utilities for preparing data for model

Adapted from: https://github.com/mexchy1000/CellDART
"""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib_venn
import matplotlib.pyplot as plt
from sklearn import preprocessing


def random_mix(X, y, nmix=5, n_samples=10000, seed=0):
    """Creates a weighted average random sampling of gene expression, and the
    corresponding weights.

    Args:
        X (:obj:, array_like of `float`): Matrix containing single cell samples
            and their GEx profiles
        y (:obj:, array_like of `int`): Array of cell type labels

    Shape:
        - X: `(N, C)`, where `N` is the number of single cell samples, and `C`
        is the number of genes
        - y: `(N,)`, where `N` is the number of single cell samples
        - pseudo_gex: :math: `(N_{out}, C)` where
        :math: `N_{out} = \text{n\_samples}` and `C` is the number of genes
        - ctps: `(N_{out}, C_{types})`
        where :math: `N_{out} = \text{n\_samples}` and :math: `C_{types}`
        is the number of cell types.

    Returns:
         - pseudo_gex (ndarray): Matrix containing pseudo-spot samples and their
         cell type proportion weighted averages
         - ctps (ndarray): Matrix containing pseudo-spot samples and their cell
         type proportions
    """
    # Define empty lists
    pseudo_gex, ctps = [], []
    ys_ = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

    rstate = np.random.RandomState(seed)
    fraction_all = rstate.rand(n_samples, nmix)
    randindex_all = rstate.randint(len(X), size=(n_samples, nmix))

    for i in range(n_samples):
        # fraction: random fraction across the "nmix" number of sampled cells
        fraction = fraction_all[i]
        fraction = fraction / np.sum(fraction)
        fraction = np.reshape(fraction, (nmix, 1))

        # Random selection of the single cell data by the index
        randindex = randindex_all[i]
        ymix = ys_[randindex]
        # Calculate the fraction of cell types in the cell mixture
        yy = np.sum(ymix * fraction, axis=0)
        # Calculate weighted gene expression of the cell mixture
        XX = np.asarray(X[randindex]) * fraction
        XX_ = np.sum(XX, axis=0)

        # Add cell type fraction & composite gene expression in the list
        ctps.append(yy)
        pseudo_gex.append(XX_)

    pseudo_gex = np.asarray(pseudo_gex)
    ctps = np.asarray(ctps)

    return pseudo_gex, ctps


def log_minmaxscale(arr):
    """returns log1pc and min/max normalized arr"""
    arrd = len(arr)
    arr = np.log1p(arr)

    arr_minus_min = arr - np.reshape(np.min(arr, axis=1), (arrd, 1))
    min2max = np.reshape((np.max(arr, axis=1) - np.min(arr, axis=1)), (arrd, 1))
    return arr_minus_min / min2max


def rank_genes(adata_sc):
    """Ranks genes for cell_subclasses.

    Args:
        adata_sc (:obj: AnnData): Single-cell data with cell_subclass.

    Returns:
        A DataFrame containing the ranked genes by cell_subclass.
    """
    sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", method="wilcoxon")

    genelists = adata_sc.uns["rank_genes_groups"]["names"]
    df_genelists = pd.DataFrame.from_records(genelists)

    return df_genelists


def select_marker_genes(
    adata_sc,
    adata_st,
    n_markers=None,
    genelists_path=None,
    force_rerank=False,
):
    """Ranks genes for cell_subclasses, finds set of top genes and selects those
    features in adatas.

    Args:
        adata_sc (:obj: AnnData): Single-cell data with cell_subclass.
        adata_st (:obj: AnnData): Spatial transcriptomic data.
        n_markers (Any): Number of top markers to include for each
            cell_subclass. If not provided, or not truthy, all genes will be
            included.
        genelists_path (Any): Path to save/load ranked genes. If not provided,
            will not save/load. Defaults to None.
        force_rerank (bool): If True, will force rerank of genes and will not
            save. Defaults to False.


    Returns:
        A tuple of a tuple of (adata_sc, adata_st) with the reduced set of
        marker genes, and a DataFrame containing the ranked genes by
        cell_subclass.
    """

    # Load or rank genes
    if force_rerank or not genelists_path:
        df_genelists = rank_genes(adata_sc)
        if genelists_path:
            df_genelists.to_pickle(genelists_path)
    else:
        try:
            df_genelists = pd.read_pickle(genelists_path)
        except FileNotFoundError as e:
            print(e)
            df_genelists = rank_genes(adata_sc)
            if genelists_path:
                df_genelists.to_pickle(genelists_path)

    all_sc_genes = set(adata_sc.var.index)
    all_st_genes = set(adata_st.var.index)

    if n_markers:
        # Get set of all top genes for cluster
        res_genes = []
        for column in df_genelists.head(n_markers):
            res_genes.extend(df_genelists.head(n_markers)[column].tolist())
        res_genes_ = list(set(res_genes))

        top_genes_sc = set(res_genes)

        # Find gene intersection
        inter_genes = [val for val in res_genes_ if val in adata_st.var.index]

        fig, ax = plt.subplots()
        matplotlib_venn.venn3_unweighted(
            [all_sc_genes, all_st_genes, top_genes_sc],
            set_labels=(
                "SC genes",
                "ST genes",
                f"Union of top {n_markers} genes for all clusters",
            ),
            ax=ax,
        )
    else:
        inter_genes = list(all_sc_genes.intersection(all_st_genes))
        fig, ax = plt.subplots()
        matplotlib_venn.venn2_unweighted(
            [all_sc_genes, all_st_genes],
            set_labels=(
                "SC genes",
                "ST genes",
            ),
            ax=ax,
        )

    print("Selected Feature Gene number", len(inter_genes))

    # Return adatas with subset of genes
    return (adata_sc[:, inter_genes], adata_st[:, inter_genes]), df_genelists, (fig, ax)
