"""Utilities for preparing data for model

Adapted from: https://github.com/mexchy1000/CellDART
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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
    ys_ = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()

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
