from math import inf

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm

SC_FILE = "../data/test.sc.h5ad"
ST_FILE = "../data/test.st.h5ad"
TREE_DEPTH = 10


def calculate_statistics(sc_data):
    cell_types = np.unique(sc_data.obs["cell_type"])
    genes = sc_data.var.index.array
    data = sc_data.X

    means = {}
    factors = {}

    for cell_type in cell_types:
        filtered = data[sc_data.obs["cell_type"] == cell_type].todense()
        totals = filtered.sum(0)
        factors[cell_type] = np.ones(shape=len(genes), dtype="float32") - (filtered.std(0) * 2)
        # factors.loc[cell_type] = np.ones(shape=len(genes)) - maxs - mins
        means[cell_type] = np.array(totals / totals.sum(), dtype="float32")

    return means, factors


# Assumes b is without error
def distance(a, b):
    scaled_diff = np.divide((a[0] - b[0]), a[1])
    return np.square(scaled_diff).sum()


def interpolate(a, b, x):
    return (
        (1 - x) * a[0] + x * b[0],
        (1 - x) * a[1] + x * b[1]
    )


def analyze(sc_data, st_data):
    sc.pp.normalize_total(st_data, target_sum=1)
    sc.pp.normalize_total(sc_data, target_sum=1)
    means, factors = calculate_statistics(sc_data)
    cells = st_data.obs.index.array
    cell_types = means.keys()
    err = 0
    total_miss = 0
    total_dist = 0

    for cell in tqdm(cells):
        excerpt = st_data[cell]
        target_counts = excerpt.X.todense().A1
        # target = (target_counts, np.zeros(target_counts.size))
        # current = (np.zeros(target_counts.size), np.zeros(target_counts.size))
        target = excerpt.X.todense().A1
        current = np.zeros(target.size)
        found_cells = []
        for step in range(TREE_DEPTH):
            best = inf
            best_coord = current
            best_type = None
            for cell_type in cell_types:
                #new_coord = interpolate(
                #    current,
                #    (means[cell_type], factors[cell_type]),
                #    1 - (step / (step + 1))
                #)
                #dist = distance(new_coord, target)
                new_coord = (step * current + means[cell_type]) / (step + 1)
                dist = np.square(np.multiply((target - new_coord), (factors[cell_type]))).sum()
                if dist < best:
                    best_type = cell_type
                    best = dist
                    best_coord = new_coord
            if best == 9001:
                break
            current = best_coord
            found_cells.append(best_type)
        [found_cell_types, counts] = np.unique(found_cells, return_counts=True)
        compare = pd.DataFrame(index=cell_types, columns=["actual", "found", "diff"])
        compare[:] = 0
        for cell_type in cell_types:
            compare.loc[cell_type, "actual"] = excerpt.obs.loc[cell, cell_type]
        compare.loc[found_cell_types, "found"] = counts
        compare["diff"] = abs(compare["actual"] - compare["found"])
        if compare["diff"].sum() > 0:
            err += 1
            total_miss += compare["diff"].sum()
            #np.array(compare["actual"], dtype="float32")
            total_dist += sp.spatial.distance.jensenshannon(
                np.array(compare["actual"], dtype="float32"),
                np.array(compare["found"], dtype="float32")
            )
            #print(f"mismatch for cell {cell}")
            #print(compare)
    print(f"Correct: {len(cells) - err}/{len(cells)}")
    print(f"Misses = {total_miss}")
    print(f"AVG Distance = {total_dist / len(cells)} (total: {total_dist})")


test_sc_data = ad.read(SC_FILE)
test_st_data = ad.read(ST_FILE)
analyze(test_sc_data, test_st_data)
