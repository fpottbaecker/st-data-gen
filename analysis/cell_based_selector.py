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
from myselectors import *

SC_FILE = "../../data/HCA_split/harvard-donor-H6-weak.sc.h5ad"
ST_FILE = "../../data/HCA_split/generated/harvard-donor-H6-weak.st.h5ad"
TREE_DEPTH = 10
ITERATION_COUNT = 2


def evaluate_jsd(actuals, predicteds):
    dists = sp.spatial.distance.jensenshannon(
        actuals,
        predicteds,
        axis=1
    )
    dists[np.isnan(dists)] = 0  # TODO: is this a good idea?
    print(f"JSD: mean={dists.mean()}, quartiles={np.quantile(dists, [0.0, 0.25, 0.5, 0.75, 1.0])}")


def evaluate_rmse(actuals, predicteds):
    squared_errors = (actuals - predicteds)**2
    print(f"RMSE: all={np.sqrt(np.nanmean(squared_errors))}, quartiles={np.quantile(np.sqrt(np.mean(squared_errors, axis=1)), [0.0, 0.25, 0.5, 0.75, 1.0])}")


def cell_based_analysis(sc_data, st_data, evaluators=(evaluate_jsd, evaluate_rmse), selector_klass=GreedyTreeSelector):
    sc.pp.normalize_total(st_data, target_sum=1)
    sc.pp.normalize_total(sc_data, target_sum=1)

    selector = selector_klass(sc_data)

    digest = util.sha256(SC_FILE)
    cache_path = f"{os.path.dirname(SC_FILE)}/.{digest}.{selector.type_name()}.pickle"
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
    reference_cell_types = pd.Index(np.unique(sc_data.obs["cell_type"]))
    actual: pd.DataFrame = st_data.obsm["Y"]
    actual_cell_types = actual.columns
    extraneous_cell_types = reference_cell_types.difference(actual_cell_types)
    actual = actual.join(pd.DataFrame(data=np.zeros(shape=(cells.size, extraneous_cell_types.size), dtype="float32"), index=actual.index, columns=extraneous_cell_types))
    predicted = pd.DataFrame(index=cells, columns=actual.columns, dtype="float32")
    hits = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")
    cumulative_hits = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")
    exceeds = np.zeros(shape=(ITERATION_COUNT, TREE_DEPTH), dtype="int32")

    for cell in tqdm(cells, ncols=80):
        excerpt = st_data[cell]
        target = selector.init_target(excerpt.X.todense().A1)
        current = selector.init_state()
        all_selected = []

        for iteration in range(ITERATION_COUNT):
            all_hit = True
            current_profile = pd.Series(data=np.zeros(reference_cell_types.size), index=reference_cell_types, dtype="float32")
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

                selected_type = selector.map_to_names(selected)
                current_profile[selected_type] += 1/TREE_DEPTH
                y_selected = actual.at[cell, selected_type]
                hit = y_selected > 0.0001
                all_hit &= hit
                if all_hit:
                    cumulative_hits[iteration, step] += 1
                if hit:
                    hits[iteration, step] += 1
                if current_profile[selected_type] > y_selected + 0.001:
                    exceeds[iteration, step] += 1

        found_cells_types = selector.map_to_names(all_selected)
        [found_cell_types, counts] = np.unique(found_cells_types, return_counts=True)
        predicted.loc[cell] = 0
        predicted.at[cell, found_cell_types] = counts / counts.sum()

    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    if callable(evaluators):
        evaluators(actual, predicted)
    else:
        for evaluator in evaluators:
            evaluator(actual, predicted)

    print(f"Step hits (%)")
    print(hits * 100.0 / cells.size)
    print(f"Full hits (%)")
    print(cumulative_hits * 100.0 / cells.size)
    print("Divergence from randomness")
    print(((hits / cells.size).cumprod(axis=1) - (cumulative_hits / cells.size)) * 100.0)
    print(f"Exceeding counts (%)")
    print(exceeds * 100.0 / cells.size)


if __name__ == "__main__":
    test_sc_data = ad.read(SC_FILE)
    test_st_data = ad.read(ST_FILE)
    cell_based_analysis(test_sc_data, test_st_data)
