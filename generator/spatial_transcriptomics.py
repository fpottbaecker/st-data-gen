from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

from util import generate_expression_profile

CELL_TYPE_FILE = "../data/test.cells.h5ad"
ST_FILE = "../data/test.st.h5ad"
NUMBER_OF_SPOTS = 1000
COUNTS_PER_SPOT = 1000
CELLS_PER_SPOT = 10


def generate(cell_spec, n_spots=NUMBER_OF_SPOTS, n_counts=COUNTS_PER_SPOT, n_cells=CELLS_PER_SPOT):
    cell_types = cell_spec.obs.index.array
    n_types = cell_types.size
    genes = cell_spec.var.index.array
    n_genes = genes.size
    cell_data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), columns=cell_types, dtype="int")
    gene_data = pd.DataFrame(index=genes, columns=[])
    data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), columns=genes)
    rng = np.random.default_rng()

    gene_p = np.ndarray(shape=(n_types, n_genes), dtype="float32")
    for cell_type in range(n_types):
        weights = cell_spec.X[cell_type]
        gene_p[cell_type] = weights / weights.sum()

    for i in tqdm(range(n_spots), desc="Generating spot data"):
        selected_cell_types = rng.choice(n_types, size=n_cells, replace=True, shuffle=False)
        cell_data.loc[i] = 0
        p = np.zeros(shape=n_genes, dtype="float64")
        for cell_type in selected_cell_types:
            p += generate_expression_profile(cell_spec, n_genes, cell_type, rng)
            cell_data.loc[i, cell_types[cell_type]] += 1
        # data.loc[i] = 0
        # recorded_genes = rng.choice(genes, size=n_counts, p=p/p.sum(), replace=True)
        # [selected_genes, values] = np.unique(recorded_genes, return_counts=True)
        # data.loc[i, selected_genes] = values
        counts = rng.multinomial(n=n_counts, pvals=p/p.sum())
        data.loc[i] = counts

    return ad.AnnData(X=data, obs=cell_data, var=gene_data)


test_spec = ad.read(CELL_TYPE_FILE)
g = generate(test_spec)
g.X = csr_matrix(g.X)
g.write(ST_FILE)
