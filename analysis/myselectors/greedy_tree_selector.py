from math import inf

import numpy as np

__all__ = ['GreedyTreeSelector']


class GreedyTreeSelector:
    means: np.ndarray
    factors: np.ndarray

    class State:
        vector: np.array
        weight: int

        def __init__(self, size):
            self.weight = 0
            self.vector = np.zeros(size)

    def __init__(self, sc_data, cell_type_var="cell_type"):
        self.sc_data = sc_data
        self.genes = sc_data.var.index
        self.n_genes = self.genes.size
        self.types = np.unique(sc_data.obs[cell_type_var])
        self.n_types = self.types.size
        self.cell_type_var = cell_type_var

    def train(self):
        self.means = np.zeros(shape=(self.n_types, self.n_genes), dtype="float32")
        self.factors = np.zeros(shape=(self.n_types, self.n_genes), dtype="float32")
        for cell_type in range(self.n_types):
            filtered = self.sc_data.X[self.sc_data.obs[self.cell_type_var] == self.types[cell_type]].todense()
            totals = filtered.sum(axis=0)
            self.means[cell_type] = totals / totals.sum()
            self.factors[cell_type] = np.ones(shape=self.n_genes, dtype="float32") - (filtered.std(0) * 2)

    def cache_data(self):
        return self.means, self.factors

    def load_cache_data(self, data):
        self.means, self.factors = data

    def init_target(self, spot_profile: np.array):
        return spot_profile

    def init_state(self):
        return GreedyTreeSelector.State(self.n_genes)

    def select_best(self, state: State, target: np.array):
        best = inf
        best_type = None
        for cell_type in range(self.n_types):
            new_coord = (state.weight * state.vector + self.means[cell_type]) / (state.weight + 1)
            dist = np.square(np.multiply((target - new_coord), (self.factors[cell_type]))).sum()
            if dist < best:
                best_type = cell_type
                best = dist
        return best_type

    def add_element(self, state, selected):
        state.vector = (state.weight * state.vector + self.means[selected]) / (state.weight + 1)
        state.weight += 1
        return state

    def remove_element(self, state, selected):
        full = state.vector * state.weight
        full -= self.means[selected]
        state.weight -= 1
        state.vector = full / state.weight
        return state

    def map_to_types(self, selected):
        return selected
