from random import shuffle, sample
from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd

NUMBER_OF_GENES = 100
NUMBER_OF_CELL_TYPES = 10
NUMBER_OF_BASELINE_GENES = 20
BASELINE_GENE_RANGE = (50, 10)
NUMBER_OF_MARKER_GENES = 8
MARKER_GENE_RANGE = (500, 100)


def generate(n_genes=NUMBER_OF_GENES, n_types=NUMBER_OF_CELL_TYPES, n_baseline=NUMBER_OF_BASELINE_GENES,
             r_baseline=BASELINE_GENE_RANGE, n_marker=NUMBER_OF_MARKER_GENES, r_marker=MARKER_GENE_RANGE):
    """
    Generate a fictional set of cell types and assign a reasonable distribution of genes to it.
    :param n_genes: the total number of genes of the "species"
    :param n_types: the number of cell types to generate
    :param n_baseline: the number of baseline genes (expressed in every cell type) to select
    :param r_baseline: the expression weight range of the baseline genes
    :param n_marker: the number of marker genes to select per cell type
    :param r_marker: the expression weight range of the marker genes
    :return an AnnData object specifying the generate cell types
    """

    rng = np.random.default_rng()

    gene_names = _generate_genes(n_genes)
    genes = pd.DataFrame(index=gene_names, columns=["baseline"])
    genes["baseline"] = 0
    genes.loc[sample(gene_names, n_baseline), "baseline"] = rng.normal(r_baseline[0], r_baseline[1], n_baseline)

    cell_names = _generate_cells(n_types)
    cells = pd.DataFrame(index=cell_names, columns=[])
    # markers = np.array(shape=(n_types, n_marker))
    weights = pd.DataFrame(index=cell_names, columns=gene_names)

    for cell in cell_names:
        weights.loc[cell] = 0
        marker_genes = sample(gene_names, n_marker)
        # cells.loc[cell, "markers"] = np.array(marker_genes)
        weights.loc[cell, marker_genes] = rng.normal(r_marker[0], r_marker[1], n_marker)
        weights.loc[cell] += genes["baseline"]

    return ad.AnnData(X=weights, obs=cells, var=genes)


def _generate_genes(n_genes):
    return [f"GENE-{i + 1}" for i in range(n_genes)]


def _generate_cells(n_cells):
    return [f"CELL-{i + 1}" for i in range(n_cells)]


g = generate()
# g.X = csr_matrix(g.X)
g.write("../data/test.cells.h5ad")
