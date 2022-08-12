from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from .matcher import Matcher


class GeneIntersectionMatcher(Matcher):
    def __init__(self):
        super().__init__()

    def match(self, reference: SingleCellDataset, target: SpatialTranscriptomicsDataset) -> float:
        sc_data = reference.anndata
        st_data = target.anndata

        sc_genes = sc_data.var_names
        st_genes = st_data.var_names

        return len(sc_genes.intersection(st_genes)) / len(sc_genes.union(st_genes))
