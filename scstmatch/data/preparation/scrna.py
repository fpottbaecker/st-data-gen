from os.path import basename, exists
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
from urllib.parse import unquote, urlparse
from urllib.request import  urlretrieve

from scstmatch.data import DatasetGroup
from .constants import SPLITTING_KEY, SPLIT_TESTING, SPLIT_TRAINING, TEMP_DIR, SOURCES_DIR
from ... import SingleCellDataset


class QualityControl:
    key: str
    starts_with: str

    max_frac: float
    def __init__(self, key, /, starts_with: str, max_frac: float = None, **kwargs):
        self.starts_with = starts_with
        self.max_frac = max_frac
        self.key = key

    def mark(self, anndata):
        anndata.var[self.key] = anndata.var_names.str.startswith(self.starts_with)

    def filter(self, anndata: ad.AnnData):
        return anndata[anndata.obs[f"pct_counts_{self.key}"] < self.max_frac * 100, :]

class Preparation:
    CELL_KEYS = ["min_counts", "max_counts", "min_genes", "max_genes"]
    cell_filters: dict[str, any]
    GENE_KEYS = ["min_counts", "max_counts", "min_cells", "max_cells"]
    gene_filters: dict[str, any]

    normalize: float
    split: tuple[float, float]

    qc_vars: dict[str, QualityControl]

    def __init__(self, /, cell_filters=None, gene_filters=None, normalize=1e6, split=None, **kwargs):
        self.cell_filters = cell_filters or {}
        self.gene_filters = gene_filters or {}
        self.split = split or [0.5, 0.5]
        self.normalize = normalize

        self.qc_vars = {key: QualityControl(key, **spec) for key, spec in cell_filters.get("quality_control", {}).items()}


    def apply(self, anndata: ad.AnnData):
        # Filter Cells and Genes
        for key in Preparation.CELL_KEYS:
            if key in self.cell_filters:
                cell_mask, _ = sc.pp.filter_cells(anndata, **{key: self.cell_filters[key]}, inplace=False)
        for key in Preparation.GENE_KEYS:
            if key in self.cell_filters:
                sc.pp.filter_cells(anndata, **{key: self.cell_filters[key]})

        # Filter for QC variables, like mito
        for qc in self.qc_vars.values():
            qc.mark(anndata)
        sc.pp.calculate_qc_metrics(anndata, qc_vars=self.qc_vars.keys(), inplace=True)
        for qc in self.qc_vars.values():
            anndata = qc.filter(anndata)


        anndata.obs[SPLITTING_KEY] = np.random.choice([SPLIT_TRAINING, SPLIT_TESTING], size=anndata.n_obs, replace=True, p=self.split)
        anndata.raw = anndata
        sc.pp.normalize_total(anndata, self.normalize)
        return anndata


class SingleCellSource:
    def __init__(self, formula, key, /, url, cell_type_column, preparation, **kwargs):
        self.key = key
        self.formula = formula
        self.url = url
        self.cell_type_column = cell_type_column
        self.preparation = Preparation(preparation)

    def setup(self):
        # TODO: Download file from url and cache data
        filename = f"{self.key}.sc.h5ad"
        source_path = SOURCES_DIR + "/" + filename
        if not exists(source_path):
            temp_path = TEMP_DIR + "/" + filename
            if not exists(temp_path):
                print(f"SOURCE: {self.key} MISSING - DOWNLOADING {basename(unquote(urlparse(self.url).path))}")
                urlretrieve(self.url, temp_path)
                print(f"SOURCE: {self.key} MISSING - DOWNLOADED AS {filename}")
            anndata = ad.read(temp_path)
            print(f"SOURCE: {self.key} MISSING - PREPARING")
            anndata = self.preparation.apply(anndata)
            print(f"SOURCE: {self.key} MISSING - WRITING")
            anndata.write(source_path)
        print(f"SOURCE: {self.key} - LOADING")
        data = SingleCellDataset.read(source_path)
        print(f"SOURCE: {self.key} - LOADED")
        data.cell_type_column = self.cell_type_column
        return DatasetGroup.from_dataset(data)
