from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from .matcher import Matcher

from sklearn.decomposition import FastICA
import numpy as np
import scanpy
import matplotlib.pyplot as plot


class DisentanglementMatcher(Matcher):
    def __init__(self):
        super().__init__()

    def match(self, reference: SingleCellDataset, target: SpatialTranscriptomicsDataset) -> float:
        sc_data = reference.anndata
        st_data = target.anndata

        scanpy.pp.pca(st_data)
        scanpy.pl.pca(st_data)
        scanpy.pl.pca_variance_ratio(st_data)

        cell_types = np.unique(sc_data.obs[reference.cell_type_column])
        n_types = len(cell_types)

        bss = FastICA(n_components=n_types, max_iter=400)
        r = bss.fit_transform(st_data.X.todense())

        result = bss.components_

        means = np.ndarray(shape=(n_types, st_data.n_vars), dtype="float64")
        for i in range(n_types):
            type = cell_types[i]
            means[i] = np.mean(sc_data[sc_data.obs.cell_type == type].X, axis=0)
        two_data = np.dot(st_data.varm["PCs"].T, result.T)[0:2, :].T
        two_data_ref = np.dot(st_data.varm["PCs"].T, means.T)[0:2, :].T
        fig = plot.figure(figsize=(6, 5), dpi=200)
        fig.subplots_adjust(left=0.1, right=0.89)
        plt = fig.add_subplot()
        plt.scatter(two_data[:, 0], two_data[:, 1])
        plt.scatter(two_data_ref[:, 0], two_data_ref[:, 1], color="red")
        fig.show()


        return np.random.uniform()
