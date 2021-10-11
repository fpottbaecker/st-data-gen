import argparse
import pathlib
from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from util import generate_expression_profile, select_cells

NUMBER_OF_SPOTS = 1000
COUNTS_PER_SPOT = 1000
CELLS_PER_SPOT = 10


def generate(cell_spec, n_spots=NUMBER_OF_SPOTS, n_counts=COUNTS_PER_SPOT, n_cells=CELLS_PER_SPOT):
    cell_types = cell_spec.obs.index.array
    n_types = cell_types.size
    genes = cell_spec.var.index.array
    n_genes = genes.size
    cell_data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), dtype="int")
    gene_data = pd.DataFrame(index=genes, columns=[])
    Y_count = np.zeros((n_spots, n_types), dtype="int32")
    data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), columns=genes)
    rng = np.random.default_rng()

    gene_p = np.ndarray(shape=(n_types, n_genes), dtype="float32")
    for cell_type in range(n_types):
        weights = cell_spec.X[cell_type]
        gene_p[cell_type] = weights / weights.sum()

    for i in tqdm(range(n_spots), desc="Generating spot data"):
        selected_cell_types = select_cells(n_cells, n_types, (i + 1) / n_spots, rng)
        cell_data.loc[i] = 0
        p = np.zeros(shape=n_genes, dtype="float64")
        for cell_type in selected_cell_types:
            p += generate_expression_profile(cell_spec, n_genes, cell_type, rng)
            Y_count[i, cell_type] += 1
        counts = rng.multinomial(n=n_counts, pvals=p/p.sum())
        data.loc[i] = counts

    Y = np.array(Y_count, dtype="float32") / Y_count.sum(axis=1)[:, np.newaxis]

    return ad.AnnData(X=data, obs=cell_data, var=gene_data,
                      obsm={
                          "Y": Y,
                          "Y_count": Y_count
                      },
                      uns={
                          "Y_labels": np.array(cell_types, dtype="str")
                      })


def main():
    parser = argparse.ArgumentParser(description="Generate a spatial transcriptomics dataset from a cell type specification.")
    parser.add_argument("-i", "--in", dest="in_file", help="The path to the input cell type specification",
                        required=True, type=pathlib.Path)
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-s", "--spots", dest="n_spots", help="The number of spots to generate",
                        type=int, default=NUMBER_OF_SPOTS)
    parser.add_argument("-c", "--counts", dest="n_counts", help="The total gene-count per spot",
                        type=int, default=COUNTS_PER_SPOT)
    parser.add_argument("-n", "--cells", dest="n_cells", help="The number of cells per spot",
                        type=int, default=CELLS_PER_SPOT)

    args = parser.parse_args()

    cell_spec = ad.read(args.in_file)
    g = generate(cell_spec, n_spots=args.n_spots, n_counts=args.n_counts, n_cells=args.n_cells)
    g.X = csr_matrix(g.X)
    g.write(args.out_file)


if __name__ == "__main__":
    main()
