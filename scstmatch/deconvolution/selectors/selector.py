from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from scstmatch.data import SingleCellDataset


class Selector(ABC):
    """
    Base class for cell picking selectors.
    """
    sc_data: SingleCellDataset

    def __init__(self):
        pass

    @abstractmethod
    def type_name(self):
        """
        :return: return a unique identifier of this selector class to use when caching
        """
        pass

    @abstractmethod
    def _load_context(self):
        """
        Load cache independent context data based on the set sc_data
        """
        pass

    @abstractmethod
    def _load_cache(self, data: Any):
        """
        Load the given cached data
        :param data: the stored cache data
        """
        pass

    @abstractmethod
    def _train(self) -> Any:
        """
        Train the selector
        :return: The data to cache for this selector
        """
        pass

    def train(self, sc_data: SingleCellDataset):
        """
        Train this selector
        :param sc_data: The reference dataset to train on
        """
        self.sc_data = sc_data
        self._load_context()
        if self.sc_data.cache[self.type_name()] is not None:
            self._load_cache(self.sc_data.cache[self.type_name()])
        else:
            self.sc_data.cache[self.type_name()] = self._train()

    @abstractmethod
    def reset(self, spot_profile: np.array):
        """
        Reset the internal state of this selector to the specified count profile
        :param spot_profile: The profile of the spot to select for
        """
        pass

    @abstractmethod
    def select_best(self) -> Any:
        """
        Identify the best option to select given the current state.
        :return: a representation of the best option
        """
        pass

    @abstractmethod
    def add_element(self, selected: Any):
        """
        Add the given option to the selection set, updating the internal state
        :param selected: The option, e.g. an option returned by `select_best`
        """
        pass

    @abstractmethod
    def remove_element(self, selected: Any):
        """
        Remove the given option from the selection set, updating the internal state
        :param selected: The option, e.g. an option returned by `select_best`
        """
        pass

    @abstractmethod
    def map_to_types(self, selected: Any):
        """
        :param selected: An option or an array of options returned by `select_best`
        :return: The cell type names of the given options
        """
        pass
