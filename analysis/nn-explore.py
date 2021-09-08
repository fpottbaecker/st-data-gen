from math import inf

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm
from scipy.spatial import KDTree

SC_FILE = "../data/train_harvard.sc.h5ad"
ST_FILE = "../data/test_harvard2.st.h5ad"
TREE_DEPTH = 10


def analyze(sc_data, st_data):
    sc.pp.normalize_total(st_data, target_sum=1)
    sc.pp.normalize_total(sc_data, target_sum=1)
    dense_data = sc_data.X.todense()
    tree = KDTree(
        data=dense_data
    )

    cells = st_data.obs.index.array
    cell_types = np.unique(sc_data.obs["cell_type"])
    err = 0
    total_miss = 0
    total_dist = 0

    for cell in tqdm(cells):
        excerpt = st_data[cell]
        target = excerpt.X.todense().A1
        current = np.zeros(target.size)
        found_cells = []
        for step in range(TREE_DEPTH):
            if step != 0:
                step_target = target + (1 / step) * (target - current)  # TODO: just try the difference vector
            else:
                step_target = target
            _, index = tree.query(step_target)
            current = (step * current + dense_data[index, :].A1) / (step + 1)
            found_cells.append(index)
        # for step in range(TREE_DEPTH):
        #    # TODO: Scaling
        #    candidate = found_cells[step]
        #    step_current = (current - (1/TREE_DEPTH) * dense_data[candidate, :].A1) * (TREE_DEPTH / TREE_DEPTH - 1)  # TODO: CHeck this
        #    step_target = target + (1 / (TREE_DEPTH - 1)) * (target - step_current)
        #    _, index = tree.query(step_target)
        #    current = ((TREE_DEPTH - 1) * step_current + dense_data[index, :].A1) / (TREE_DEPTH)
        #    found_cells[step] = index
        found_cells_types = [sc_data.obs["cell_type"][index] for index in found_cells]
        [found_cell_types, counts] = np.unique(found_cells_types, return_counts=True)
        ref = pd.Series(index=st_data.uns["Y_labels"], dtype="int32")
        ref[:] = 0
        ref[found_cell_types] = counts
        found = np.array(ref, dtype="float32")
        #total_miss += compare["diff"].sum()
        total_dist += sp.spatial.distance.jensenshannon(
            excerpt.obsm["Y"][0],
            found / found.sum()
        )
        #print(f"mismatch for cell {cell}")
        #print(compare)
    print(f"Correct: {len(cells) - err}/{len(cells)}")
    print(f"Misses = {total_miss}")
    print(f"AVG Distance = {total_dist / len(cells)} (total: {total_dist})")


test_sc_data = ad.read(SC_FILE)
test_st_data = ad.read(ST_FILE)
analyze(test_sc_data, test_st_data)
