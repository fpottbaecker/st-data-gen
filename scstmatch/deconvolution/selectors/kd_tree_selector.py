import numpy as np
from scipy.spatial import KDTree

__all__ = ['KDTreeSelector']

from .selector import Selector


class KDTreeSelector(Selector):
    tree: KDTree
    # State variables
    target_profile: np.ndarray
    num_selected: int
    current_profile: np.ndarray

    def __init__(self):
        super().__init__()

    def type_name(self):
        return KDTreeSelector.__name__

    def _load_context(self):
        anndata = self.sc_data.anndata
        self.dense_data = anndata.X.todense()
        self.genes = anndata.var.index
        self.n_genes = self.genes.size
        self.cell_type_var = self.sc_data.cell_type_column
        self.types = np.unique(anndata.obs[self.cell_type_var])
        self.n_types = self.types.size

    def _train(self):
        self.tree = KDTree(data=self.dense_data, copy_data=True)
        return self.tree

    def _load_cache(self, data):
        self.tree = data

    def reset(self, spot_profile: np.array):
        self.target_profile = spot_profile
        self.num_selected = 0
        self.current_profile = np.zeros(self.n_genes)

    def select_best(self):
        if self.num_selected == 0:
            step_target = self.target_profile
        else:
            step_target = self.target_profile + (1 / self.num_selected) * (self.target_profile - self.current_profile)

        _, index = self.tree.query(step_target)
        return index

    def add_element(self, selected):
        self.current_profile = (self.current_profile * self.num_selected + self.dense_data[selected, :].A1) / (
                    self.num_selected + 1)
        self.num_selected += 1

    def remove_element(self, selected):
        full = self.current_profile * self.num_selected
        full -= self.dense_data[selected, :].A1
        self.num_selected -= 1
        self.current_profile = full / self.num_selected

    def map_to_types(self, selected):
        return self.sc_data.anndata.obs[self.cell_type_var][selected]
