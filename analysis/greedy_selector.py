import os.path
from math import inf

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm
from scipy.spatial import KDTree
import util
from myselectors import KDTreeSelector

SC_FILE = "../data/train_harvard.sc.h5ad"
ST_FILE = "../data/test_harvard2.st.h5ad"
TREE_DEPTH = 10


def evaluate_jsd(actuals, predicteds):
    dists = sp.spatial.distance.jensenshannon(
        actuals,
        predicteds,
        axis=1
    )
    print(f"JSD: mean={dists.mean()}, quartiles=({np.quantile(dists, 0)}, {np.quantile(dists, 0.25)}, [mean], {np.quantile(dists, 0.75)}, {np.quantile(dists, 1)})")


def analyze(sc_data, st_data, evaluators=evaluate_jsd):
    sc.pp.normalize_total(st_data, target_sum=1)
    sc.pp.normalize_total(sc_data, target_sum=1)

    selector = KDTreeSelector(sc_data)

    digest = util.sha256(SC_FILE)
    cache_path = f"{os.path.dirname(SC_FILE)}/.{digest}.{type(selector).__name__}.pickle"
    if os.path.exists(cache_path):
        print("using cached training data")
        cache_data = util.load_pickle(cache_path)
        selector.load_cache_data(cache_data)
    else:
        print("training anew")
        selector.train()
        print("caching training data")
        util.pickle_to_file(selector.cache_data(), cache_path)

    cells = st_data.obs.index.array
    cell_types = np.unique(sc_data.obs["cell_type"])
    actual = st_data.obsm["Y"]
    predicted = np.ndarray(shape=actual.shape, dtype="float32")
    cell_index = 0

    for cell in tqdm(cells):
        excerpt = st_data[cell]
        target = selector.init_target(excerpt.X.todense().A1)
        current = selector.init_sate()
        all_selected = []

        for step in range(TREE_DEPTH):
            selected = selector.select_best(current, target)
            all_selected.append(selected)
            current = selector.add_element(current, selected)

        #for _ in range(1):
        #    for step in range(TREE_DEPTH):
        #        # TODO: Scaling
        #        candidate = all_selected[step]
        #        current = selector.remove_element(current, candidate)
        #        selected = selector.select_best(current, target)
        #        all_selected[step] = selected
        #        current = selector.add_element(current, selected)

        found_cells_types = selector.map_to_types(all_selected)
        [found_cell_types, counts] = np.unique(found_cells_types, return_counts=True)
        ref = pd.Series(index=st_data.uns["Y_labels"], dtype="int32")
        ref[:] = 0
        ref[found_cell_types] = counts
        found = np.array(ref, dtype="float32")
        predicted[cell_index, :] = found / found.sum()
        cell_index += 1

    if callable(evaluators):
        evaluators(actual, predicted)
    else:
        for evaluator in evaluators:
            evaluator(actual, predicted)


test_sc_data = ad.read(SC_FILE)
test_st_data = ad.read(ST_FILE)
analyze(test_sc_data, test_st_data)
