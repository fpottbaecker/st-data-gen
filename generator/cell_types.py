from random import shuffle, sample
import argparse
import pathlib
from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd

NUMBER_OF_GENES = 500
NUMBER_OF_CELL_TYPES = 10
NUMBER_OF_BASELINE_GENES = 200
BASELINE_GENE_RANGE = (500, 100)
BASELINE_GENE_STD_RANGE = (500, 100)
NUMBER_OF_MARKER_GENES = 20
MARKER_GENE_RANGE = (500, 100)
MARKER_GENE_STD_RANGE = (50, 10)


def generate(n_genes=NUMBER_OF_GENES, n_types=NUMBER_OF_CELL_TYPES, n_baseline=NUMBER_OF_BASELINE_GENES,
             r_baseline=BASELINE_GENE_RANGE, std_baseline=BASELINE_GENE_STD_RANGE, n_marker=NUMBER_OF_MARKER_GENES,
             r_marker=MARKER_GENE_RANGE, std_marker=MARKER_GENE_STD_RANGE):
    """
    Generate a fictional set of cell types and assign a reasonable distribution of genes to it.
    :param n_genes: the total number of genes of the "species"
    :param n_types: the number of cell types to generate
    :param n_baseline: the number of baseline genes (expressed in every cell type) to select
    :param r_baseline: the expression weight range of the baseline genes
    :param std_baseline: the expression weight standard deviation range of baseline genes
    :param n_marker: the number of marker genes to select per cell type
    :param r_marker: the expression weight range of the marker genes
    :param std_marker: the expression weight standard deviation range of marker genes
    :return an AnnData object specifying the generate cell types
    """

    rng = np.random.default_rng()

    gene_names = _generate_genes(n_genes)
    genes = pd.DataFrame(index=gene_names, columns=["baseline", "baseline_std"])
    genes["baseline"] = 0
    genes["baseline_std"] = 0
    baseline_genes = sample(gene_names, n_baseline)
    genes.loc[baseline_genes, "baseline"] = rng.normal(r_baseline[0], r_baseline[1], n_baseline)
    genes.loc[baseline_genes, "baseline_std"] = rng.normal(std_baseline[0], std_baseline[1], n_baseline)

    cell_names = _generate_cells(n_types)
    cells = pd.DataFrame(index=cell_names, columns=[])
    # markers = np.array(shape=(n_types, n_marker))
    weights = pd.DataFrame(index=cell_names, columns=gene_names)
    std = pd.DataFrame(index=cell_names, columns=gene_names)

    for cell in cell_names:
        weights.loc[cell] = 0
        std.loc[cell] = 0
        marker_genes = sample(gene_names, n_marker)
        # cells.loc[cell, "markers"] = np.array(marker_genes)
        weights.loc[cell, marker_genes] = rng.normal(r_marker[0], r_marker[1], n_marker)
        std.loc[cell, marker_genes] = rng.normal(std_marker[0], std_marker[1], n_marker)
        weights.loc[cell] += genes["baseline"]
        std.loc[cell] += genes["baseline_std"]

    data = ad.AnnData(X=weights, obs=cells, var=genes)
    data.layers["std"] = std.to_numpy(dtype="float32")
    return data


def _generate_genes(n_genes):
    return [f"GENE-{i + 1}" for i in range(n_genes)]


def _generate_cells(n_cells):
    return [f"CELL-{i + 1}" for i in range(n_cells)]


def main():
    parser = argparse.ArgumentParser(description="Generate cell type specifications and export them.")
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-t", "--types", dest="n_types", help="The number of cell_types to generate",
                        type=int, default=NUMBER_OF_CELL_TYPES)
    parser.add_argument("-g", "--genes", dest="n_genes", help="The number of genes to generate",
                        type=int, default=NUMBER_OF_GENES)
    parser.add_argument("-m", "--markers", dest="n_marker", help="The number of marker genes per cell type",
                        type=int, default=NUMBER_OF_MARKER_GENES)
    parser.add_argument("-b", "--baseline", dest="n_baseline", help="The number of baseline genes across all cell types",
                        type=int, default=NUMBER_OF_BASELINE_GENES)

    args = parser.parse_args()
    g = generate(n_genes=args.n_genes, n_types=args.n_types, n_marker=args.n_marker, n_baseline=args.n_baseline)
    g.write(args.out_file)


if __name__ == "__main__":
    main()
