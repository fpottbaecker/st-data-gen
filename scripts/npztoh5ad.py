import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

NPZ_FILE = "../data/Sanger_Nuclei_small_test.npz"
AD_FILE = "../data/Sanger_Nuclei_small_test.st.h5ad"
TREE_DEPTH = 10


def convert_ad(npz):
    y = pd.DataFrame(npz["ST_Y_test"], columns=npz["cell_types"])
    data = ad.AnnData(X=npz["ST_X_test"], var=pd.DataFrame(index=npz["genes"]))
    y.index = data.obs_names
    data.obsm["Y"] = y

    return data


npz_data = np.load(NPZ_FILE)
ad_data = convert_ad(npz_data)
ad_data.X = csr_matrix(ad_data.X)
ad_data.write(AD_FILE)
