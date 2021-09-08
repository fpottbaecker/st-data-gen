from math import inf
from scipy.sparse import csr_matrix

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
from tqdm import tqdm
from scipy.spatial import KDTree

NPZ_FILE = "../data/test_harvard.st.npz"
AD_FILE = "../data/test_harvard.st.h5ad"
TREE_DEPTH = 10

def convert_ad(npz):
    return ad.AnnData(X=npz["ST_X_test"], var=pd.DataFrame(index=npz["genes"]),
                      obsm={
                          "Y": npz["ST_Y_test"]
                      },
                      uns={
                          "Y_labels": npz["cell_types"]
                      })


npz_data = np.load(NPZ_FILE)
ad_data = convert_ad(npz_data)
ad_data.X = csr_matrix(ad_data.X)
ad_data.write(AD_FILE)
