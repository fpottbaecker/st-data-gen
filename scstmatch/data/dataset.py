import os.path
from scstmatch.util import load_pickle, pickle_to_file, sha256_for_file


class Dataset:
    def __init__(self, source_path=None):
        self.source_path = source_path
        self.cache = DatasetCache(self)

    def _write(self, path):
        pass

    def write(self, path):
        self._write(path)
        self.source_path = path


class DatasetCache:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def source_path(self):
        if not self.dataset.source_path:
            raise ValueError("Cannot generate cache path for in memory dataset")
        return self.dataset.source_path

    def path_for(self, key):
        dirname = os.path.dirname(self.source_path())
        filename = os.path.basename(self.source_path())
        return f"{dirname}/.caches/{filename}.{sha256_for_file(self.source_path())}/{key}.pickle"

    def __getitem__(self, key):
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        return load_pickle(path)

    def __setitem__(self, key, value):
        path = self.path_for(key)
        pickle_to_file(value, path)
