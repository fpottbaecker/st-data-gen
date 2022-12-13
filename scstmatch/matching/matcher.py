from abc import ABC, abstractmethod
from typing import Any

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class Matcher(ABC):
    """
    Base class for data matching methods
    """
    reference: SingleCellDataset
    is_trained: bool

    def __init__(self, reference: SingleCellDataset):
        """
        :param reference: The reference to match target dataset against.
        """
        self.reference = reference
        self.is_trained = False

    def _train(self):
        """
        Perform training for the reference dataset
        """
        pass

    @abstractmethod
    def _match(self, target: SpatialTranscriptomicsDataset) -> Any:
        """
        Perform matching of the reference and target dataset
        :param target: the target dataset
        :return: a scoring value, either per spot or per dataset
        """
        pass

    def train(self):
        """
        Train the matcher if it is not trained yet
        """
        if self.is_trained:
            return
        self._train()
        self.is_trained = True

    def match(self, target: SpatialTranscriptomicsDataset) -> Any:
        """
        Evaluate the match quality of the reference and target dataset
        :param target: the target dataset
        :return: a scoring value, either per spot or per dataset
        """
        self.train()
        return self._match(target)
