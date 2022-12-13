from abc import ABC, abstractmethod

import pandas as pd

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class Deconvolver(ABC):
    """
    Base class for deconvolution methods
    """
    reference: SingleCellDataset
    is_trained: bool

    def __init__(self, reference: SingleCellDataset):
        """
        :param reference: the reference set to train with
        """
        self.reference = reference
        self.is_trained = False

    def _train(self):
        """
        Perform training for the current reference dataset, implementation is optional
        """
        pass

    @abstractmethod
    def _deconvolve(self, target: SpatialTranscriptomicsDataset) -> pd.DataFrame:
        """
        :param target: An ST dataset to deconvolve
        :return: The per spot profile mixtures predicted
        """
        pass

    def train(self):
        """
        Train this deconvolver if it has not been trained yet
        """
        if self.is_trained:
            return
        self._train()
        self.is_trained = True

    def deconvolve(self, target: SpatialTranscriptomicsDataset) -> pd.DataFrame:
        """
        Perform deconvolution on a target dataset
        :param target: The target dataset to deconvolve
        :return: A data frame indexed by `target.anndata.obs_names`, columns are the cell types present in the reference dataset
        """
        self.train()
        return self._deconvolve(target)
