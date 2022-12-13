from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from .matcher import Matcher


class GeneIntersectionMatcher(Matcher):
    """
    Prototypical matcher using the gene set overlap
    """
    def __init__(self, reference: SingleCellDataset):
        super().__init__(reference)

    def _match(self, target: SpatialTranscriptomicsDataset) -> float:
        sc_data = self.reference.anndata
        st_data = target.anndata

        # TODO: identify HVG or marker genes
        sc_genes = sc_data.var_names
        st_genes = st_data.var_names

        return len(sc_genes.intersection(st_genes)) / len(sc_genes.union(st_genes))
