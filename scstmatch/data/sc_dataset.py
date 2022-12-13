from os.path import splitext

import anndata as ad

from .dataset import Dataset

DEFAULT_CELL_TYPE_COLUMN = "cell_type"


class SingleCellDataset(Dataset):
    cell_type_column: str

    def __init__(self, anndata: ad.AnnData, path: str = None):
        super().__init__(anndata, path)
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

    def copy_with(self, anndata: ad.AnnData):
        c = SingleCellDataset(anndata, self.source_path)
        c.cell_type_column = self.cell_type_column
        return c
