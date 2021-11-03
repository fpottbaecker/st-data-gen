import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
import pandas as pd

__all__ = ['PCAWrapper', 'wrap_pca']

NUM_PCS = 50

def wrap_pca(selector_klass):
    def wrap(sc_data):
        return PCAWrapper(sc_data, selector_klass)
    return wrap


class PCAWrapper:
    selector: object

    def __init__(self, sc_data, selector_klass, cell_type_var="cell_type"):
        self.sc_data = sc_data
        sc.pp.pca(sc_data)
        self.pca_data = ad.AnnData(X=sc_data.obsm["X_pca"][:, 0:NUM_PCS], var=pd.DataFrame(index=range(NUM_PCS)), obs=sc_data.obs)
        self.pca_data.X = csr_matrix(self.pca_data.X)
        self.gene_matrix = sc_data.varm["PCs"][:, 0:NUM_PCS].T
        self.genes = sc_data.var.index
        self.n_genes = self.genes.size
        self.types = np.unique(sc_data.obs[cell_type_var])
        self.n_types = self.types.size
        self.cell_type_var = cell_type_var
        self.selector = selector_klass(self.pca_data)

    def type_name(self):
        return f"{PCAWrapper.__name__}.{self.selector.type_name()}"

    def train(self):
        self.selector.train()

    def cache_data(self):
        return self.selector.cache_data()

    def load_cache_data(self, data):
        self.selector.load_cache_data(data)

    def init_target(self, spot_profile: np.array):
        return self.gene_matrix @ spot_profile

    def init_state(self):
        return self.selector.init_state()

    def select_best(self, state, target: np.array):
        return self.selector.select_best(state, target)

    def add_element(self, state, selected):
        return self.selector.add_element(state, selected)

    def remove_element(self, state, selected):
        return self.selector.remove_element(state, selected)

    def map_to_types(self, selected):
        return self.selector.map_to_types(selected)
