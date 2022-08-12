import numpy as np

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class Deconvolver:
    reference: SingleCellDataset

    def __init__(self, reference: SingleCellDataset):
        self.reference = reference

    def _train(self):
        pass

    def _deconvolve(self, st_data: SpatialTranscriptomicsDataset) -> np.array:
        pass

    def deconvolve(self, st_data: SpatialTranscriptomicsDataset) -> np.array:
        return self._deconvolve(st_data)
