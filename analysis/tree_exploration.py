import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

SC_FILE = "../data/test.sc.h5ad"
ST_FILE = "../data/test.st.h5ad"
TREE_DEPTH = 10


def calculate_statistics(sc_data):
    cell_types = np.unique(sc_data.obs["cell_type"])
    genes = sc_data.var.index.array
    data = sc_data.X

    means = pd.DataFrame(index=cell_types, columns=genes)
    factors = pd.DataFrame(index=cell_types, columns=genes)

    for cell_type in cell_types:
        filtered = data[sc_data.obs["cell_type"] == cell_type].todense()
        totals = filtered.sum(0)
        factors.loc[cell_type] = np.ones(shape=len(genes)) - (filtered.std(0) * 2)
        # factors.loc[cell_type] = np.ones(shape=len(genes)) - maxs - mins
        means.loc[cell_type] = totals / totals.sum()

    return means, factors


def analyze(sc_data, st_data):
    sc.pp.normalize_total(st_data, target_sum=1)
    sc.pp.normalize_total(sc_data, target_sum=1)
    means, factors = calculate_statistics(sc_data)
    cells = st_data.obs.index.array
    cell_types = means.index.array
    err = 0

    for cell in tqdm(cells):
        excerpt = st_data[cell]
        target = excerpt.X.todense().A1
        current = np.zeros(target.size)
        found_cells = []
        for step in range(TREE_DEPTH):
            best = 9001
            best_coord = current
            best_type = None
            for cell_type in cell_types:
                new_coord = (step * current + means.loc[cell_type]) / (step + 1)
                dist = (((target - new_coord) * (factors.loc[cell_type])) ** 2).sum()
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
            #print(f"mismatch for cell {cell}")
            #print(compare)
    print(f"Correct: {len(cells) - err}/{len(cells)}")


test_sc_data = ad.read(SC_FILE)
test_st_data = ad.read(ST_FILE)
analyze(test_sc_data, test_st_data)
