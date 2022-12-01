
from os.path import basename

from scstmatch.matching import *
from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset

#SC_FILE = "../../data/HCA_split/harvard-donor-H6.sc.h5ad"
SC_FILE = "../data/reference/hca_harvard_gender_Male.sc.h5ad"
#ST_FILE = "../../data/HCA_split/generated/harvard-donor-H6-weak.st.h5ad"
# ST_FILE = "../data/synthetic_HCA1.st.h5ad"
ST_FILE = "../data/st/hca_harvard_donor_H6.st.h5ad"


def matching_test():
    matcher = SpotNMatch()
    print(matcher.match(SingleCellDataset.read(SC_FILE), SpatialTranscriptomicsDataset.read(ST_FILE)))
    print(f"{basename(SC_FILE)} => {basename(ST_FILE)}")


if __name__ == "__main__":
    matching_test()
