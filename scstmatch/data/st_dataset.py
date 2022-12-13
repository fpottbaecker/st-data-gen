from os.path import splitext

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .dataset import Dataset


class SpatialTranscriptomicsDataset(Dataset):
    def __init__(self, anndata: ad.AnnData, path: str = None):
        super().__init__(anndata, path)

    @staticmethod
    def read_anndata(path: str):
        return SpatialTranscriptomicsDataset(ad.read(path), path)

    @staticmethod
    def read_npz(path: str, materialize=False):
        npz = np.load(path)
        anndata = ad.AnnData(X=npz["ST_X_test"], var=pd.DataFrame(index=npz["genes"]))
        anndata.obsm["Y"] = pd.DataFrame(npz["ST_Y_test"], index=anndata.obs_names, columns=npz["cell_types"])
        anndata.X = csr_matrix(anndata.X)
        if materialize:
            anndata.write(path + ".h5ad")
        return SpatialTranscriptomicsDataset(anndata, path if not materialize else path + ".h5ad")

    @staticmethod
    def read(path: str):
        return {
            ".h5ad": SpatialTranscriptomicsDataset.read_anndata,
            ".npz": SpatialTranscriptomicsDataset.read_npz
        }[splitext(path)[1]](path)

    def _write(self, path):
        self.anndata.write(path)
