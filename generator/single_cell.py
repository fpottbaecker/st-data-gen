import argparse
import pathlib
from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from util import generate_expression_profile

NUMBER_OF_CELLS = 10000
COUNTS_PER_CELL = 1000


def generate(cell_spec, n_cells=NUMBER_OF_CELLS, n_counts=COUNTS_PER_CELL):
    cell_types = cell_spec.obs.index.array
    n_types = cell_types.size
    genes = cell_spec.var.index.array
    n_genes = genes.size
    cell_data = pd.DataFrame(index=pd.RangeIndex(0, n_cells), columns=["cell_type"])
    gene_data = pd.DataFrame(index=genes, columns=[])
    data = pd.DataFrame(index=pd.RangeIndex(0, n_cells), columns=genes)
    rng = np.random.default_rng()

    for i in tqdm(range(n_cells), desc="Generating cell data"):
        cell_type = rng.choice(n_types)
        gene_p = generate_expression_profile(cell_spec, n_genes, cell_type, rng)
        cell_data.loc[i, "cell_type"] = cell_types[cell_type]
        counts = rng.multinomial(n=n_counts, pvals=gene_p)
        data.loc[i] = counts

    return ad.AnnData(X=data, obs=cell_data, var=gene_data)


def main():
    parser = argparse.ArgumentParser(description="Generate a single cell dataset from a cell type specification.")
    parser.add_argument("-i", "--in", dest="in_file", help="The path to the input cell type specification",
                        required=True, type=pathlib.Path)
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-s", "--samples", dest="n_cells", help="The number of (cell)samples to generate",
                        type=int, default=NUMBER_OF_CELLS)
    parser.add_argument("-c", "--counts", dest="n_counts", help="The total gene-count per cell",
                        type=int, default=COUNTS_PER_CELL)

    args = parser.parse_args()

    cell_spec = ad.read(args.in_file)
    g = generate(cell_spec, n_cells=args.n_cells, n_counts=args.n_counts)
    g.X = csr_matrix(g.X)
    g.write(args.out_file)


if __name__ == "__main__":
    main()
