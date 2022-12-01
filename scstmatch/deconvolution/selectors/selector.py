from typing import Any

import numpy as np

from scstmatch.data import SingleCellDataset


class Selector:
    sc_data: SingleCellDataset

    def __init__(self):
        pass

    def type_name(self):
        pass

    def _load_context(self):
        pass

    def _load_cache(self, data: Any):
        pass

    def _train(self) -> Any:
        pass

    def train(self, sc_data: SingleCellDataset):
        self.sc_data = sc_data
        self._load_context()
        if self.sc_data.cache[self.type_name()] is not None:
            self._load_cache(self.sc_data.cache[self.type_name()])
        else:
            self.sc_data.cache[self.type_name()] = self._train()

    def reset(self, spot_profile: np.array):
        pass

    def select_best(self) -> Any:
        pass

    def add_element(self, selected: Any):
        pass

    def remove_element(self, selected: Any):
        pass

    def map_to_types(self, selected: Any):
        pass
