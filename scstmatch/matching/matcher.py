from typing import Any

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class Matcher():
    reference: SingleCellDataset
    is_trained: bool

    def __init__(self, reference: SingleCellDataset):
        self.reference = reference
        self.is_trained = False

    def _train(self):
        pass

    def _match(self, target: SpatialTranscriptomicsDataset) -> Any:
        pass

    def train(self):
        if self.is_trained:
            return
        self._train()
        self.is_trained = True

    def match(self, target: SpatialTranscriptomicsDataset) -> Any:
        self.train()
        return self._match(target)
