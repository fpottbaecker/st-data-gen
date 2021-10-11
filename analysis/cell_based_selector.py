import os.path
from math import inf

import anndata as ad
import numpy
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm
from scipy.spatial import KDTree
import util
from myselectors import *

SC_FILE = "../data/test2.sc.h5ad"
ST_FILE = "../data/test2.st.h5ad"
TREE_DEPTH = 10
ITERATION_COUNT = 4


def evaluate_jsd(actuals, predicteds):
    dists = sp.spatial.distance.jensenshannon(
        actuals,
        predicteds,
        axis=1
    )
    dists[np.isnan(dists)] = 0  # TODO: is this a good idea?
    print(f"JSD: mean={dists.mean()}, quartiles={np.quantile(dists, [0.0, 0.25, 0.5, 0.75, 1.0])}")


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
    hits = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")
    cumulative_hits = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")
    exceeds = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")

    for cell in tqdm(cells):
        excerpt = st_data[cell]
        target = selector.init_target(excerpt.X.todense().A1)
        current = selector.init_state()
        all_selected = []

        for iteration in range(ITERATION_COUNT):
            all_hit = True
            current_profile = np.zeros(shape=cell_types.size, dtype="float32")
            for step in range(TREE_DEPTH):
                if iteration != 0:
                    candidate = all_selected[step]
                    current = selector.remove_element(current, candidate)

                selected = selector.select_best(current, target)

                if iteration == 0:
                    all_selected.append(selected)
                else:
                    all_selected[step] = selected
                current = selector.add_element(current, selected)

                selected_type = selector.map_to_types([selected])[0]
                current_profile[selected_type] += 1/TREE_DEPTH
                all_hit &= excerpt.obsm["Y"][0, selected_type] > 0.0001
                if all_hit:
                    cumulative_hits[iteration, step] += 1
                if excerpt.obsm["Y"][0, selected_type] > 0.0001:
                    hits[iteration, step] += 1
                if current_profile[selected_type] > excerpt.obsm["Y"][0, selected_type] + 0.001:
                    exceeds[iteration, step] += 1

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

    numpy.set_printoptions(precision=4, linewidth=200, suppress=True)
    print(f"Step hits (%)")
    print(hits * 100.0 / cells.size)
    print(f"Full hits (%)")
    print(cumulative_hits * 100.0 / cells.size)
    print("Divergence from randomness")
    print(((hits / cells.size).cumprod(axis=1) - (cumulative_hits / cells.size)) * 100.0)
    print(f"Exceeding counts (%)")
    print(exceeds * 100.0 / cells.size)


test_sc_data = ad.read(SC_FILE)
test_st_data = ad.read(ST_FILE)
analyze(test_sc_data, test_st_data)