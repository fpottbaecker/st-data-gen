import pandas as pd

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class Deconvolver:
    reference: SingleCellDataset
    is_trained: bool

    def __init__(self, reference: SingleCellDataset):
        self.reference = reference
        self.is_trained = False

    def _train(self):
        pass

    def _deconvolve(self, target: SpatialTranscriptomicsDataset) -> pd.DataFrame:
        pass

    def train(self):
        if self.is_trained:
            return
        self._train()
        self.is_trained = True

    def deconvolve(self, target: SpatialTranscriptomicsDataset) -> pd.DataFrame:
        self.train()
        return self._deconvolve(target)
