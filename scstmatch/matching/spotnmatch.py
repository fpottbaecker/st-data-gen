import numpy as np
import scanpy
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.util import Timer
from .matcher import Matcher


class SpotNMatch(Matcher):
    """
    Implementation of the SPotNMatch data matching method
    """
    def __init__(self, reference: SingleCellDataset):
        super().__init__(reference)

    def _train(self):
        timer = Timer()
        sc_data = self.reference.anndata
        cell_types = np.unique(sc_data.obs[self.reference.cell_type_column])
        # TODO: Normalization maybe?
        V = sc_data.X.T
        H = np.zeros(shape=(len(cell_types), sc_data.n_obs), dtype="float32")
        W = None
        timer.restart("init train")
        if self.reference.cache["SPOTLightMatcher"] is None:
            # Seed respective topics
            for ct in cell_types:
                H[np.where(cell_types == ct), sc_data.obs[self.reference.cell_type_column] == ct] = 1
            timer.restart("seed H")
            sc_data.layers["log1p"] = scanpy.pp.normalize_total(sc_data, inplace=False, target_sum=1e6)["X"]
            scanpy.pp.log1p(sc_data, layer="log1p")
            # Identify marker genes and seed genes
            scanpy.tl.rank_genes_groups(sc_data, groupby=self.reference.cell_type_column, use_raw=False, layer="log1p",
                                        method="t-test", key_added="ranks")
            timer.restart("rank genes")
            is_marker = np.full(sc_data.n_vars, False)
            for ct in cell_types:
                markers = sc_data.uns["ranks"]["names"][ct][sc_data.uns["ranks"]["scores"][ct] > 25]
                is_marker[sc_data.var_names.get_indexer(markers)] |= True

            self.all_markers = sc_data.var_names[is_marker]
            W = np.zeros(shape=(len(self.all_markers), len(cell_types)), dtype="float32")

            for ct in cell_types:
                all_scores = sc_data.uns["ranks"]["scores"][ct]
                min_score = 100
                relevant = all_scores > min_score
                n_top = 50
                relevant[0:n_top] = True
                markers = sc_data.uns["ranks"]["names"][ct][relevant]
                scores = all_scores[relevant]
                adjusted = (scores / np.max(scores)) * 1e6
                W[self.all_markers.get_indexer(markers), np.where(cell_types == ct)] = adjusted
            timer.restart("seed W")
            nmf = NMF(len(cell_types), init="custom", max_iter=500, tol=1e-2, solver="cd")
            self.W = nmf.fit_transform(X=V[is_marker, :], W=W, H=H)
            H = nmf.components_
            timer.restart(f"NMF ({nmf.n_iter_} it)")
            self.reference.cache["SPOTLightMatcher"] = (self.all_markers, self.W)
            timer.restart("save")
        else:
            self.all_markers, self.W = self.reference.cache["SPOTLightMatcher"]
            timer.restart("load")

    def _match(self, target: SpatialTranscriptomicsDataset) -> float:
        timer = Timer()
        sc_data = self.reference.anndata
        st_data = target.anndata
        cell_types = np.unique(sc_data.obs[self.reference.cell_type_column])

        if "log1p" not in st_data.layers:
            st_data.layers["log1p"] = scanpy.pp.normalize_total(st_data, inplace=False, target_sum=1000)["X"]
            scanpy.pp.log1p(st_data, layer="log1p")
        if "highly_variable" not in st_data.var:
            scanpy.pp.highly_variable_genes(st_data, layer="log1p", flavor="seurat")
            timer.restart("hvg")
        st_markers = np.isin(st_data.var_names, self.all_markers)
        # find H', such that V_ = W x H'
        V_ = st_data.X.T[st_markers, :]
        # filter before evaluation
        vV_ = st_data.X[:, np.array(st_data.var.highly_variable) & st_markers].T

        residuals = np.zeros(st_data.n_obs)
        solutions = np.zeros(shape=(st_data.n_obs, len(cell_types)))

        # filter before NNLS fit
        xresiduals = np.zeros(st_data.n_obs)
        xsolutions = np.zeros(shape=(st_data.n_obs, len(cell_types)))

        highly_variable_markers = np.isin(self.all_markers, st_data.var_names[st_data.var.highly_variable])

        totals = V_.power(2).sum(axis=0).A1
        vtotals = vV_.power(2).sum(axis=0).A1
        for i in range(st_data.n_obs):
            solutions[i], residuals[i] = nnls(self.W, V_.getcol(i).A.flatten())
            xsolutions[i], xresiduals[i] = nnls(self.W[highly_variable_markers], vV_.getcol(i).A.flatten())
        timer.restart("nnls")

        prediction = np.dot(self.W, solutions.T)

        residuals = np.sum((prediction - V_).A ** 2, axis=0)  # none
        vresiduals = np.sum((prediction[highly_variable_markers] - vV_).A ** 2, axis=0)  # evaluate
        xresiduals = np.sum(((np.dot(self.W[highly_variable_markers], xsolutions.T) - vV_).A ** 2), axis=0)  # fit

        unexplained_residuals = 1 - (residuals / totals)
        vunexplained_residuals = 1 - (vresiduals / vtotals)
        xunexplained_residuals = 1 - (xresiduals / vtotals)


        vresult = 1 - (np.sum(vresiduals) / np.sum(vtotals))
        xresult = 1 - (np.sum(xresiduals) / np.sum(vtotals))
        result = 1 - (np.sum(residuals) / np.sum(totals))
        # print(result)

        # NOTE: This is the extended return value setup, which reduced training and runtime for full evaluation.
        # return vresult, xresult, result, vunexplained_residuals, xunexplained_residuals, unexplained_residuals
        return vunexplained_residuals
