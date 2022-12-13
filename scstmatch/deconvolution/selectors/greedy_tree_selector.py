from math import inf

import numpy as np
import scanpy as sc

__all__ = ['GreedySelector']

from .selector import Selector


class GreedySelector(Selector):
    means: np.ndarray
    factors: np.array
    # State variables
    target_profile: np.ndarray
    num_selected: int
    current_profile: np.ndarray

    def __init__(self):
        super().__init__()

    def type_name(self):
        return GreedySelector.__name__

    def _load_context(self):
        anndata = self.sc_data.anndata
        self.genes = anndata.var.index
        self.n_genes = self.genes.size
        cell_type_var = self.sc_data.cell_type_column
        self.types = np.unique(anndata.obs[cell_type_var])
        self.n_types = self.types.size

    def _train(self):
        anndata = self.sc_data.anndata
        cell_type_var = self.sc_data.cell_type_column
        self.means = np.zeros(shape=(self.n_types, self.n_genes), dtype="float32")
        self.factors = np.zeros(shape=(self.n_types, self.n_genes), dtype="float32")
        if "norm" not in anndata.layers:
            anndata.layers["norm"] = sc.pp.normalize_total(anndata, target_sum=1, inplace=False)["X"]

        for cell_type in range(self.n_types):
            filtered = anndata.layers["norm"][anndata.obs[cell_type_var] == self.types[cell_type]].todense()
            totals = filtered.sum(axis=0)
            self.means[cell_type] = totals / totals.sum()
            self.factors[cell_type] = np.ones(shape=self.n_genes, dtype="float32") - (filtered.std(0) * 2)

        return self.means, self.factors

    def _load_cache(self, data):
        self.means, self.factors = data

    def reset(self, spot_profile: np.array):
        self.target_profile = spot_profile
        self.num_selected = 0
        self.current_profile = np.zeros(self.n_genes)

    def select_best(self):
        best = inf
        best_type = None
        for cell_type in range(self.n_types):
            new_coord = (self.num_selected * self.current_profile + self.means[cell_type]) / (self.num_selected + 1)
            dist = np.square(np.multiply((self.target_profile - new_coord), (self.factors[cell_type]))).sum()
            if dist < best:
                best_type = cell_type
                best = dist
        return best_type

    def add_element(self, selected):
        self.current_profile = (self.num_selected * self.current_profile + self.means[selected].flatten()) / (
                    self.num_selected + 1)
        self.num_selected += 1

    def remove_element(self, selected):
        full = self.current_profile * self.num_selected
        full -= self.means[selected]
        self.num_selected -= 1
        self.current_profile = full / self.num_selected

    def map_to_types(self, selected):
        return self.types[selected]
