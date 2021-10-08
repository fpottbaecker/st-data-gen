import numpy as np
from scipy.spatial import KDTree

__all__ = ['KDTreeSelector']


class KDTreeSelector:
    tree: KDTree

    class State:
        vector: np.array
        weight: int

        def __init__(self, size):
            self.weight = 0
            self.vector = np.zeros(size)

    def __init__(self, sc_data, cell_type_var="cell_type"):
        self.sc_data = sc_data
        self.dense_data = self.sc_data.X.todense()
        self.genes = sc_data.var.index
        self.n_genes = self.genes.size
        self.types = np.unique(sc_data.obs[cell_type_var])
        self.n_types = self.types.size
        self.cell_type_var = cell_type_var

    def train(self):
        self.tree = KDTree(data=self.dense_data, copy_data=True)

    def cache_data(self):
        return self.tree

    def load_cache_data(self, data):
        self.tree = data

    def init_target(self, spot_profile: np.array):
        return spot_profile

    def init_state(self):
        return KDTreeSelector.State(self.n_genes)

    def select_best(self, state: State, target: np.array):
        if state.weight == 0:
            step_target = target
        else:
            step_target = target + (1 / state.weight) * (target - state.vector)

        _, index = self.tree.query(step_target)
        return index

    def add_element(self, state, selected):
        state.vector = (state.vector * state.weight + self.dense_data[selected, :].A1) / (state.weight + 1)
        state.weight += 1
        return state

    def remove_element(self, state, selected):
        full = state.vector * state.weight
        full -= self.dense_data[selected, :].A1
        state.weight -= 1
        state.vector = full / state.weight
        return state

    def map_to_types(self, selected):
        return [np.where(self.types == self.sc_data.obs[self.cell_type_var][index]) for index in selected]
