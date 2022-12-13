import anndata as ad
import numpy as np
import pandas as pd

from scstmatch.data import CellTypeDataset
from .generator import Generator


class CellTypeGenerator(Generator):
    """
    Generator for synthetic tissue profiles
    """
    rng: np.random.Generator

    def __init__(self, **options):
        super().__init__(defaults=dict(
            n_genes=500,
            n_cells=10,
            n_baseline=200,
            dist_baseline=(500, 100),
            dist_baseline_std=(500, 100),
            n_marker=20,
            dist_marker=(500, 100),
            dist_marker_std=(50, 10),
        ), **options)

    def _update_inputs(self):
        self.rng = np.random.default_rng()

    def _generate(self) -> CellTypeDataset:
        gene_names = [f"GENE-{i + 1}" for i in range(self.options.n_genes)]

        genes = pd.DataFrame(index=gene_names, columns=["baseline", "baseline_std"])
        genes["baseline"] = 0
        genes["baseline_std"] = 0

        baseline_genes = self.rng.choice(gene_names, self.options.n_baseline)
        dist = self.options.dist_baseline
        genes.loc[baseline_genes, "baseline"] = self.rng.normal(dist[0], dist[1], self.options.n_baseline)
        dist = self.options.dist_baseline_std
        genes.loc[baseline_genes, "baseline_std"] = self.rng.normal(dist[0], dist[1], self.options.n_baseline)

        cell_names = [f"CELL-{i + 1}" for i in range(self.options.n_cells)]
        cells = pd.DataFrame(index=cell_names, columns=[])
        # markers = np.array(shape=(self.options.n_types, self.options.n_marker))
        weights = pd.DataFrame(index=cell_names, columns=gene_names)
        std = pd.DataFrame(index=cell_names, columns=gene_names)

        for cell in cell_names:
            weights.loc[cell] = 0
            std.loc[cell] = 0

            marker_genes = self.rng.choice(gene_names, self.options.n_marker, replace=False)
            # cells.loc[cell, "markers"] = np.array(marker_genes)
            dist = self.options.dist_marker
            weights.loc[cell, marker_genes] = self.rng.normal(dist[0], dist[1], self.options.n_marker)
            dist = self.options.dist_marker_std
            std.loc[cell, marker_genes] = self.rng.normal(dist[0], dist[1], self.options.n_marker)
            weights.loc[cell] += genes["baseline"]
            std.loc[cell] += genes["baseline_std"]

        data = ad.AnnData(X=weights, obs=cells, var=genes)
        data.layers["std"] = std.to_numpy(dtype="float32")

        return CellTypeDataset(data)
