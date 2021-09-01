from math import inf

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm
from scipy.spatial import KDTree

SC_FILE = "../data/test2.sc.h5ad"
ST_FILE = "../data/test2.st.h5ad"
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
                step_target = target + (1 / step) * (target - current)
            else:
                step_target = target
            _, index = tree.query(step_target)
            current = (step * current + dense_data[index, :].A1) / (step + 1)
            found_cells.append(sc_data.obs["cell_type"][index])
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
