
from scstmatch.data import SingleCellDataset
from .deconvolver import Deconvolver


class GreedySelector(Deconvolver):

    def __init__(self, reference: SingleCellDataset):
        super().__init__(reference)
        self._train()

    def _train(self):
        pass

