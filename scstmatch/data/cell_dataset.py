import anndata as ad
from os.path import splitext

from .dataset import Dataset


class CellTypeDataset(Dataset):
    def __init__(self, anndata: ad.AnnData, path: str = None):
        super().__init__(anndata, path)

    @staticmethod
    def read_anndata(path: str):
        return CellTypeDataset(ad.read(path), path)

    @staticmethod
    def read(path: str):
        return {
            ".h5ad": CellTypeDataset.read_anndata,
        }[splitext(path)[1]](path)

    def copy_with(self, anndata: ad.AnnData):
        return CellTypeDataset(anndata, self.source_path)
