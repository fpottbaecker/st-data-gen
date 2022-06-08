import anndata as ad
from os.path import splitext

from .dataset import Dataset

DEFAULT_CELL_TYPE_COLUMN = "cell_type"


class SingleCellDataset(Dataset):
    anndata: ad.AnnData
    cell_type_column: str

    def __init__(self, anndata: ad.AnnData, path: str = None):
        super().__init__(path)
        self.anndata = anndata
        if DEFAULT_CELL_TYPE_COLUMN in self.anndata.obs:
            self.cell_type_column = DEFAULT_CELL_TYPE_COLUMN

    @staticmethod
    def read_anndata(path: str):
        return SingleCellDataset(ad.read(path), path)

    @staticmethod
    def read(path: str):
        return {
            ".h5ad": SingleCellDataset.read_anndata,
        }[splitext(path)[1]](path)

    def _write(self, path):
        self.anndata.write(path)
