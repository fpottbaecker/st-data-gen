import io
import yaml
from os.path import exists
from os import mkdir

from .scrna import SingleCellSource
from .variant import DatasetVariant
from .constants import TEMP_DIR, SOURCES_DIR, REFERENCE_DIR, ST_DIR


class DatasetFormula:
    single_cell_sources: dict[str, SingleCellSource]
    variants: dict[str, DatasetVariant]

    def __init__(self, path):
        with io.open(path, "r") as file:
            self.data = yaml.safe_load(file)

        self.single_cell_sources = {key: SingleCellSource(self, key, **data) for key, data in self.data["sources"]["scrna"].items()}
        self.variants = {key: DatasetVariant(self, **data) for key, data in self.data["variants"].items()}

    def setup(self):
        for path in [TEMP_DIR, SOURCES_DIR, REFERENCE_DIR, ST_DIR]:
            if not exists(path):
                mkdir(path)

        for key, variant in self.variants.items():
            print(f"SETUP VARIANT: {key}")
            variant.setup()
            print(f"SETUP VARIANT: {key} DONE")

