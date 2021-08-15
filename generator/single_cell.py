from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from util import generate_expression_profile

CELL_TYPE_FILE = "../data/test.cells.h5ad"
SC_FILE = "../data/test.sc.h5ad"
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


test_spec = ad.read(CELL_TYPE_FILE)
g = generate(test_spec)
g.X = csr_matrix(g.X)
g.write(SC_FILE)
