import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

from scstmatch.data import SingleCellDataset
from scstmatch.data import SpatialTranscriptomicsDataset
from .deconvolver import Deconvolver
from .selectors.selector import Selector


class IntegralDeconvolver(Deconvolver):
    iteration_count: int
    tree_depth: int
    debug: bool

    selector: Selector

    def __init__(self, reference: SingleCellDataset, selector: Selector, iteration_count=2, tree_depth=10, debug=False):
        super().__init__(reference)
        self.selector = selector
        self.iteration_count = iteration_count
        self.tree_depth = tree_depth
        self.debug = debug

    def _train(self):
        self.selector.train(self.reference)
        self.reference_cell_types = pd.Index(np.unique(self.reference.anndata.obs[self.reference.cell_type_column]))

    def selected_to_profile(self, selected, absolute=True):
        cell_types = self.selector.map_to_types(selected)
        [cell_types, counts] = np.unique(cell_types, return_counts=True)
        predicted = pd.Series(0, index=self.reference_cell_types, dtype="float32")
        factor = self.tree_depth if absolute else counts.sum()
        predicted.loc[cell_types] = counts / factor
        return predicted

    def _deconvolve(self, target: SpatialTranscriptomicsDataset) -> pd.DataFrame:

        st_data = target.anndata
        if "norm" not in st_data.layers:
            st_data.layers["norm"] = sc.pp.normalize_total(st_data, target_sum=1, inplace=False)["X"]
        spots = st_data.obs.index.array

        # Deal with extraneous / missing cell types
        actual: pd.DataFrame = st_data.obsm["Y"]
        actual_cell_types = actual.columns
        extraneous_cell_types = self.reference_cell_types.difference(actual_cell_types)
        actual = actual.join(
            pd.DataFrame(data=np.zeros(shape=(spots.size, extraneous_cell_types.size), dtype="float32"),
                         index=actual.index, columns=extraneous_cell_types))
        predicted = pd.DataFrame(0, index=spots, columns=actual.columns, dtype="float32")

        # Debugging variables
        debug_shape = (self.iteration_count, self.tree_depth)
        hits = np.zeros(shape=debug_shape, dtype="int32")
        cumulative_hits = np.zeros(shape=debug_shape, dtype="int32")
        exceeds = np.zeros(shape=debug_shape, dtype="int32")

        for spot in tqdm(spots, ncols=80):
            excerpt = st_data[spot]
            self.selector.reset(excerpt.layers["norm"].todense().A1)
            all_selected = []

            for iteration in range(self.iteration_count):
                all_hit = True
                for step in range(self.tree_depth):
                    if iteration != 0:
                        candidate = all_selected[step]
                        self.selector.remove_element(candidate)

                    selected = self.selector.select_best()
                    if iteration == 0:
                        all_selected.append(selected)
                    else:
                        all_selected[step] = selected
                    self.selector.add_element(selected)

                    selected_type = self.selector.map_to_types(selected)
                    current_profile = self.selected_to_profile(all_selected)
                    y_selected = actual.at[spot, selected_type]
                    hit = y_selected > 0.0001
                    all_hit &= hit
                    if all_hit:
                        cumulative_hits[iteration, step] += 1
                    if hit:
                        hits[iteration, step] += 1
                    if current_profile[selected_type] > y_selected + 0.001:
                        exceeds[iteration, step] += 1

            predicted.at[spot, self.reference_cell_types] = self.selected_to_profile(all_selected, False)

        if self.debug:
            print(f"Step hits (%)")
            print(hits * 100.0 / spots.size)
            print(f"Full hits (%)")
            print(cumulative_hits * 100.0 / spots.size)
            print("Divergence from randomness")
            print(((hits / spots.size).cumprod(axis=1) - (cumulative_hits / spots.size)) * 100.0)
            print(f"Exceeding counts (%)")
            print(exceeds * 100.0 / spots.size)

        return predicted
