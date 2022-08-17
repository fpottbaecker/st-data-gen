import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

from .generator import Generator
from .util import generate_expression_profile, select_cells
from scstmatch.data import CellTypeDataset, SpatialTranscriptomicsDataset


class SpatialTranscriptomicsGenerator(Generator):
    rng: np.random.Generator

    def __init__(self, cell_spec: CellTypeDataset, **options):
        super().__init__(
            inputs={"cell_spec": cell_spec},
            defaults=dict(
                n_spots=1000,
                n_counts=1000,
                n_cells=10,
            ), **options)

    def _update_inputs(self):
        self.rng = np.random.default_rng()

    def _generate(self) -> SpatialTranscriptomicsDataset:
        # TODO: Review generation procedure
        cell_spec = self.inputs.cell_spec.anndata
        cell_types = cell_spec.obs.index.array
        n_types = cell_types.size
        genes = cell_spec.var.index.array
        n_genes = genes.size
        cell_data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_spots), dtype="int")
        gene_data = pd.DataFrame(index=genes, columns=[])
        y_counts = pd.DataFrame(data=np.zeros(shape=(self.options.n_spots, n_types)), index=cell_data.index, columns=cell_types, dtype="int32")
        data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_spots), columns=genes)

        gene_p = np.ndarray(shape=(n_types, n_genes), dtype="float32")
        for cell_type in range(n_types):
            weights = cell_spec.X[cell_type]
            gene_p[cell_type] = weights / weights.sum()

        for i in tqdm(range(self.options.n_spots), desc="Generating spot data"):
            selected_cell_types = select_cells(self.options.n_cells, n_types, (i + 1) / self.options.n_spots, self.rng)
            cell_data.loc[i] = 0
            p = np.zeros(shape=n_genes, dtype="float64")
            for cell_type in selected_cell_types:
                p += generate_expression_profile(cell_spec, n_genes, cell_type, self.rng)
                y_counts.at[i, cell_types[cell_type]] += 1
            counts = self.rng.multinomial(n=self.options.n_counts, pvals=p / p.sum())
            data.loc[i] = counts

        y = pd.DataFrame(data=y_counts / np.array(y_counts.sum(axis=1))[:, np.newaxis], dtype="float32")

        anndata = ad.AnnData(X=data, obs=cell_data, var=gene_data)

        y.index = anndata.obs_names
        anndata.obsm["Y"] = y
        y_counts.index = anndata.obs_names
        anndata.obsm["Y_counts"] = y_counts

        anndata.X = csr_matrix(anndata.X)
        return SpatialTranscriptomicsDataset(anndata)
