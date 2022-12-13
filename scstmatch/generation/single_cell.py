import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from scstmatch.data import CellTypeDataset, SingleCellDataset
from .generator import Generator
from .util import generate_expression_profile


class SingleCellGenerator(Generator):
    rng: np.random.Generator

    def __init__(self, cell_spec: CellTypeDataset, **options):
        super().__init__(
            inputs={"cell_spec": cell_spec},
            defaults=dict(
                n_samples=10000,
                n_counts=1000,
            ),
            **options)

    def _update_inputs(self):
        self.rng = np.random.default_rng()

    def _generate(self) -> SingleCellDataset:
        # TODO: Review generation procedure
        cell_spec = self.inputs.cell_spec.anndata
        cell_types = cell_spec.obs.index.array
        n_types = cell_types.size
        genes = cell_spec.var.index.array
        n_genes = genes.size
        cell_data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_samples), columns=["cell_type"])
        gene_data = pd.DataFrame(index=genes, columns=[])
        data = pd.DataFrame(index=pd.RangeIndex(0, self.options.n_samples), columns=genes)

        for i in tqdm(range(self.options.n_samples), desc="Generating cell data"):  # TODO: Remove progressbar here?
            cell_type = self.rng.choice(n_types)
            gene_p = generate_expression_profile(cell_spec, n_genes, cell_type, self.rng)
            cell_data.loc[i, "cell_type"] = cell_types[cell_type]
            counts = self.rng.multinomial(n=self.options.n_counts, pvals=gene_p)
            data.loc[i] = counts

        anndata = ad.AnnData(X=data, obs=cell_data, var=gene_data)
        anndata.X = csr_matrix(anndata.X)
        return SingleCellDataset(anndata)
