import scanpy
import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.util import Timer
from .matcher import Matcher


class SPOTLightMatcher(Matcher):
    def __init__(self):
        super().__init__()

    def match(self, reference: SingleCellDataset, target: SpatialTranscriptomicsDataset) -> float:
        timer = Timer()
        sc_data = reference.anndata
        st_data = target.anndata

        cell_types = np.unique(sc_data.obs[reference.cell_type_column])
        # TODO: Normalization maybe?
        V = sc_data.X.T
        H = np.zeros(shape=(len(cell_types), sc_data.n_obs), dtype="float32")
        W = np.zeros(shape=(sc_data.n_vars, len(cell_types)), dtype="float32")
        timer.stop("init")
        if reference.cache["SPOTLightMatcher"] is None:
            # Seed respective topics
            for ct in cell_types:
                H[np.where(cell_types == ct), sc_data.obs[reference.cell_type_column] == ct] = 1
            timer.stop("seed H")
            # Identify marker genes and seed genes
            scanpy.tl.rank_genes_groups(sc_data, groupby=reference.cell_type_column, method="t-test", key_added="ranks")
            timer.stop("rank genes")
            for ct in cell_types:
                all_scores = sc_data.uns["ranks"]["scores"][ct]
                relevant = all_scores > 5
                markers = sc_data.uns["ranks"]["names"][ct][relevant]
                scores = all_scores[relevant]
                adjusted = scores / np.max(scores)
                W[sc_data.var_names.get_indexer(markers), np.where(cell_types == ct)] = adjusted
            timer.stop("seed W")
            nmf = NMF(len(cell_types), init="custom")
            W = nmf.fit_transform(X=V, W=W, H=H)
            H = nmf.components_
            timer.stop("NMF")
            reference.cache["SPOTLightMatcher"] = W
            timer.stop("save")
        else:
            W = reference.cache["SPOTLightMatcher"]
            timer.stop("load")

        st_data.layers["log1p"] = scanpy.pp.normalize_total(st_data, inplace=False, target_sum=1000)["X"]
        scanpy.pp.log1p(st_data, layer="log1p")
        scanpy.pp.highly_variable_genes(st_data, layer="log1p", flavor="seurat")
        timer.stop("hvg")
        # find H', such that V_ = W x H'
        V_ = st_data.X.T
        vV_ = st_data.X[:, st_data.var.highly_variable].T

        residuals = np.zeros(st_data.n_obs)
        solutions = np.zeros(shape=(st_data.n_obs, len(cell_types)))

        xresiduals = np.zeros(st_data.n_obs)
        xsolutions = np.zeros(shape=(st_data.n_obs, len(cell_types)))

        totals = V_.power(2).sum(axis=0).A1
        vtotals = vV_.power(2).sum(axis=0).A1
        for i in range(st_data.n_obs):
            solutions[i], residuals[i] = nnls(W, V_.getcol(i).A.flatten())
            xsolutions[i], xresiduals[i] = nnls(W[st_data.var.highly_variable], vV_.getcol(i).A.flatten())
        timer.stop("nnls")

        prediction = np.dot(W, solutions.T)

        residuals = np.sum((prediction - V_).A ** 2, axis=0)
        vresiduals = np.sum((prediction[st_data.var.highly_variable] - vV_).A ** 2, axis=0)
        xresiduals = np.sum(((np.dot(W[st_data.var.highly_variable], xsolutions.T) - vV_).A ** 2), axis=0)

        unexplained_residuals = 1 - (residuals / totals)
        vunexplained_residuals = 1 - (vresiduals / vtotals)
        xunexplained_residuals = 1 - (xresiduals / vtotals)

        print(f"1) {vunexplained_residuals.tolist()}")
        print(1 - (np.sum(vresiduals) / np.sum(vtotals)))

        print(f"2) {xunexplained_residuals.tolist()}")
        print(1 - (np.sum(xresiduals) / np.sum(vtotals)))

        print(f"old) {unexplained_residuals.tolist()}")

        return 1 - (np.sum(residuals) / np.sum(totals))
