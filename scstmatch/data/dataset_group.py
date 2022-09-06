
from collections.abc import Callable
from anndata import AnnData

from .dataset import Dataset


class DatasetGroup:
    data: dict[str, Dataset]

    def __init__(self, data: dict[str, Dataset]):
        self.data = data

    @classmethod
    def from_dataset(cls, dataset):
        return DatasetGroup({"": dataset})

    @staticmethod
    def join_keys(a, b):
        return a + ("_" if a != "" and b != "" else "") + b

    def map(self, fn: Callable[[Dataset], Dataset]):
        return DatasetGroup({key: fn(value) for key, value in self.data.items()})

    def split(self, fn: Callable[[str, Dataset], dict[str, Dataset]]):
        return DatasetGroup({k: v for key, value in self.data.items() for k, v in fn(key, value).items()})
