from .action import Action
from scstmatch.data.preparation.constants import REFERENCE_DIR, SPLITTING_KEY, SPLIT_TRAINING


class OutputReference(Action):
    def __init__(self, /, **kwargs):
        super().__init__()

    def perform(self, dataset_group):
        def write_reference(key, dataset):
            ad = dataset.anndata
            dataset = dataset.copy_with(ad[ad.obs[SPLITTING_KEY] == SPLIT_TRAINING, :])
            dataset.write(REFERENCE_DIR + "/" + key + ".sc.h5ad")
        dataset_group.iter(write_reference)
        return dataset_group
