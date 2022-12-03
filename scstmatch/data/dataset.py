import os.path

import anndata as ad

from scstmatch.util import load_pickle, pickle_to_file, sha256_for_file


class Dataset:
    anndata: ad.AnnData

    def __init__(self, anndata: ad.AnnData, source_path=None):
        self.anndata = anndata
        self.source_path = source_path
        self.cache = DatasetCache(self)

    def _write(self, path):
        self.anndata.write(path)

    def write(self, path):
        self._write(path)
        self.source_path = path

    def copy_with(self, anndata: ad.AnnData):
        return Dataset(anndata, self.source_path)


class DatasetCache:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def source_path(self):
        if not self.dataset.source_path:
            raise ValueError("Cannot generate cache path for in memory dataset")
        return self.dataset.source_path

    def directory(self):
        dirname = os.path.dirname(self.source_path())
        filename = os.path.basename(self.source_path())
        return f"{dirname}/.caches/{filename}.{sha256_for_file(self.source_path())}"

    def path_for(self, key):
        dirname = os.path.dirname(self.source_path())
        filename = os.path.basename(self.source_path())
        return f"{self.directory()}/{key}.pickle"

    def __getitem__(self, key):
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        return load_pickle(path)

    def __setitem__(self, key, value):
        path = self.path_for(key)
        pickle_to_file(value, path)
