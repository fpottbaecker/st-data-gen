import os.path

import anndata as ad

from scstmatch.util import load_pickle, pickle_to_file, sha256_for_file


class Dataset:
    """
    A generic transcriptomics count dataset.

    Attributes:
        anndata: The underlying annotated count data
        cache: The cache utility for this dataset
        source_path: The location of the file on disk, if persisted
    """
    anndata: ad.AnnData

    def __init__(self, anndata: ad.AnnData, source_path: str = None):
        """
        Create a generic dataset
        :param anndata: the AnnData object to represent
        :param source_path: the path where the AnnData object was loaded from
        """
        self.anndata = anndata
        self.source_path = source_path
        self.cache = DatasetCache(self)

    def _write(self, path):
        self.anndata.write(path)

    def write(self, path):
        """
        Writes the dataset to disk, this also sets the source_path accordingly
        :param path: the path to write to
        """
        self._write(path)
        self.source_path = path

    def copy_with(self, anndata: ad.AnnData):
        """
        :param anndata: An annotated count dataset
        :return: A copy of this dataset proxy with the given anndata
        """
        return Dataset(anndata, self.source_path)


class DatasetCache:
    """
    A cache utility for dataset classes
    """
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
        """
        :param key: A cache key
        :return: The cached data, if it exists
        """
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        return load_pickle(path)

    def __setitem__(self, key, value):
        """
        Stores the given object in the cache, which writes to disk
        :param key: A cache key
        :param value: The data to cache
        """
        path = self.path_for(key)
        pickle_to_file(value, path)
