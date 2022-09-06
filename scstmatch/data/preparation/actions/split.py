import numpy as np

from scstmatch.data import DatasetGroup
from .action import Action


class SplitAction(Action):
    by: str

    def __init__(self, /, by: str, **kwargs):
        super().__init__(**kwargs)
        self.by = by

    def perform(self, dataset_group: DatasetGroup):
        def dataset_split(parent_key, dataset):
            ad = dataset.anndata
            return {
                DatasetGroup.join_keys(parent_key, self.by + "_" + str(value)): dataset.copy_with(ad[ad.obs[self.by] == value])
                for value in np.unique(ad.obs[self.by])}

        return dataset_group.split(dataset_split)
