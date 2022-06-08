import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

from .generator import Generator
from .util import select_cells
from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset


class SC2STGenerator(Generator):
    rng: np.random.Generator

    def __init__(self, sc_data: SingleCellDataset, **options):
        super().__init__(
            inputs={"sc_data": sc_data},
            defaults=dict(
                n_spots=1000,
                n_cells=10,
            ), **options)

    def _update_inputs(self):
        self.rng = np.random.default_rng()
        sc_data = self.inputs.sc_data.anndata
        cell_type_column = self.inputs.sc_data.cell_type_column
        cell_types = np.unique(sc_data.obs[cell_type_column])
        self.counts_cache = {}
        for cell_type in cell_types:
            self.counts_cache[cell_type] = sc_data[sc_data.obs[cell_type_column] == cell_type, :].X.toarray()

    def _generate(self) -> SpatialTranscriptomicsDataset:
        sc_data = self.inputs.sc_data.anndata
        cell_type_column = self.inputs.sc_data.cell_type_column
        cell_types = np.unique(sc_data.obs[cell_type_column])
        n_types = cell_types.size
        genes = sc_data.var.index.array
        cell_data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_spots), dtype="int")
        gene_data = pd.DataFrame(index=genes, columns=[])
        y_count = np.zeros(shape=(self.options.n_spots, n_types), dtype="int32")
        data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_spots), columns=genes)
        rng = np.random.default_rng()

        for i in tqdm(range(self.options.n_spots), desc="Generating spot data"):
            selected_cell_types = select_cells(self.options.n_cells, n_types, (i + 1) / self.options.n_spots, rng)
            cell_data.loc[i] = 0
            data.loc[i] = 0
            for cell_type in selected_cell_types:
                cell_type_name = cell_types[cell_type]
                data.loc[i] += rng.choice(self.counts_cache[cell_type_name])
                y_count[i, cell_type] += 1

        y = np.array(y_count, dtype="float32") / y_count.sum(axis=1)[:, np.newaxis]

        anndata = ad.AnnData(X=data, obs=cell_data, var=gene_data,
                             obsm={
                                 "Y": y,
                                 "Y_count": y_count
                             },
                             uns={
                                 "Y_labels": np.array(cell_types, dtype="str")
                             })
        anndata.X = csr_matrix(anndata.X)
        return SpatialTranscriptomicsDataset(anndata)
