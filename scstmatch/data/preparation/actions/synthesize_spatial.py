from .action import Action
from scstmatch.data.preparation.constants import ST_DIR, SPLITTING_KEY, SPLIT_TESTING

from scstmatch.generation import SC2STGenerator


class SynthesizeSpatial(Action):
    counts: tuple[int, int]
    cells: tuple[int, int]

    def __init__(self, /, counts=None, cells=None, **kwargs):
        super().__init__()
        self.counts = (counts[0], counts[1]) if cells is not None else (10000, 10000)
        self.cells = (cells[0], cells[1]) if cells is not None else (10, 10)

    def perform(self, dataset_group):
        def generate_st(key, dataset):
            ad = dataset.anndata
            dataset = dataset.copy_with(ad[ad.obs[SPLITTING_KEY] == SPLIT_TESTING, :])
            generator = SC2STGenerator(dataset)
            gen = generator.generate()
            gen.write(ST_DIR + "/" + key + ".st.h5ad")
        dataset_group.iter(generate_st)
        return dataset_group
