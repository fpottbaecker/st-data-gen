from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

SC_FILE = "../data/donor_h5.sc.h5ad"
ST_FILE = "../data/donor_h5.st.h5ad"
CELL_TYPE_COLUMN = "cell_type"
NUMBER_OF_SPOTS = 1000
CELLS_PER_SPOT = 10  # TODO: Maybe have a range here


def generate(sc_data, n_spots=NUMBER_OF_SPOTS, n_cells=CELLS_PER_SPOT):
    cell_types = np.unique(sc_data.obs[CELL_TYPE_COLUMN])
    n_types = cell_types.size
    genes = sc_data.var.index.array
    n_genes = genes.size
    cell_data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), columns=cell_types, dtype="int")
    gene_data = pd.DataFrame(index=genes, columns=[])
    data = pd.DataFrame(index=pd.RangeIndex(0, n_spots), columns=genes)
    rng = np.random.default_rng()
    cell_types_cache = {}
    desified_x_cache = {}
    for cell_type in cell_types:
        cell_types_cache[cell_type] = sc_data[sc_data.obs[CELL_TYPE_COLUMN] == cell_type, :]
        desified_x_cache[cell_type] = cell_types_cache[cell_type].X.toarray()

    for i in tqdm(range(n_spots), desc="Generating spot data"):
        selected_cell_types = rng.choice(cell_types, size=n_cells, replace=True, shuffle=False)
        cell_data.loc[i] = 0
        data.loc[i] = 0
        for cell_type in selected_cell_types:
            index = rng.choice(cell_types_cache[cell_type].n_obs)
            data.loc[i] += desified_x_cache[cell_type][index, :]
            cell_data.loc[i, cell_type] += 1

    return ad.AnnData(X=data, obs=cell_data, var=gene_data)


test_sc = ad.read(SC_FILE)
g = generate(test_sc)
g.X = csr_matrix(g.X)
g.write(ST_FILE)

