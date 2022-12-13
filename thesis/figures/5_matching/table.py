import multiprocessing
from functools import partial

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.matching import SpotNMatch

# NOTE: This File requires a modification of the SpotNMatch class to return all values, not just the v scores

FILE_A = "../data/reference/hca_sanger_gender_Female.sc.h5ad"
FILE_B = "../data/reference/hca_harvard_gender_Male_-muscles.sc.h5ad"

COLUMNS = {
    ("Harvard", "full", "F"): "../data/reference/hca_harvard_gender_Female.sc.h5ad",
    ("Harvard", "full", "M"): "../data/reference/hca_harvard_gender_Male.sc.h5ad",
    ("Harvard", "V1", "F"): "../data/reference/hca_harvard_gender_Female_-muscles.sc.h5ad",
    ("Harvard", "V1", "M"): "../data/reference/hca_harvard_gender_Male_-muscles.sc.h5ad",
    ("Harvard", "V2", "F"): "../data/reference/hca_harvard_gender_Female_-endothelial.sc.h5ad",
    ("Harvard", "V2", "M"): "../data/reference/hca_harvard_gender_Male_-endothelial.sc.h5ad",
    ("Sanger", "full", "F"): "../data/reference/hca_sanger_gender_Female.sc.h5ad",
    ("Sanger", "full", "M"): "../data/reference/hca_sanger_gender_Male.sc.h5ad",
    ("Sanger", "V1", "M"): "../data/reference/hca_sanger_gender_Male_-muscles.sc.h5ad",
    # ("Harvard", "full", "H6"): "../data/reference/hca_harvard_donor_H6.sc.h5ad",
    # ("Sanger", "full", "D2"): "../data/reference/hca_sanger_donor_D2.sc.h5ad",
    # "EVERYTHING": "../data/.sources/hca.sc.h5ad",
}

ROWS = {
    ("Harvard", "full", "F"): "../data/st/hca_harvard_donor_H6.st.h5ad",
    ("Harvard", "full", "M"): "../data/st/hca_harvard_donor_H3.st.h5ad",
    ("Harvard", "V1", "F"): "../data/st/hca_harvard_donor_H6_-muscles.st.h5ad",
    ("Harvard", "V1", "M"): "../data/st/hca_harvard_donor_H3_-muscles.st.h5ad",
    ("Harvard", "V2", "F"): "../data/st/hca_harvard_donor_H6_-endothelial.st.h5ad",
    ("Harvard", "V2", "M"): "../data/st/hca_harvard_donor_H3_-endothelial.st.h5ad",
    ("Sanger", "full", "F"): "../data/st/hca_sanger_donor_D5.st.h5ad",
    ("Sanger", "full", "M"): "../data/st/hca_sanger_donor_D2.st.h5ad",
}


def generate_cell(column, row):
    column, column_path = column
    row, row_path = row
    print(f"BEGIN {column}, {row}")
    reference = SingleCellDataset.read(column_path)
    target = SpatialTranscriptomicsDataset.read(row_path)
    result = SpotNMatch().match(reference, target)
    print(f"END {column}, {row}")
    return column, row, result


def generate_column(rows, head):
    header, head_path = head
    print(f"BEGIN {header}")
    reference = SingleCellDataset.read(head_path)
    matcher = SpotNMatch()
    result = []
    for _, path in rows.items():
        target = SpatialTranscriptomicsDataset.read(path)
        result.append(matcher.match(reference, target))
        del target
    print(f"END {header}")
    del reference
    return result


def generate_table(rows, columns):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        data = np.array(p.map(partial(generate_column, rows), columns.items()))

        df_evaluate = pd.DataFrame(data[:, :, 0], columns=rows.keys(), index=columns.keys())
        df_fit = pd.DataFrame(data[:, :, 1], columns=rows.keys(), index=columns.keys())
        df = pd.DataFrame(data[:, :, 2], columns=rows.keys(), index=columns.keys())
        # data = p.starmap(generate_cell, itertools.product(columns.items(), rows.items()))
        df.index.name = "Target"
        df.columns.name = "Reference"

    print(df_evaluate.style.highlight_max(color="black", props="bfseries:").set_precision(4).to_latex(hrules=True,
                                                                                             clines="skip-last;index"))


def generate_data():
    pairs = {
        "match": (("Harvard", "full", "F"), ("Harvard", "full", "F")),
        "mismatch": (("Sanger", "V1", "M"), ("Harvard", "full", "F")),
        # "H6": (("Harvard", "full", "H6"), ("Harvard", "full", "F")),
        # "D2": (("Sanger", "full", "D2"), ("Harvard", "full", "F")),
    }

    results = {}
    vresults = {}

    plot.rcParams.update({
        "text.usetex": True,
    })

    for label, paired in pairs.items():
        print(label)
        reference = SingleCellDataset.read(COLUMNS[paired[0]])
        target = SpatialTranscriptomicsDataset.read(ROWS[paired[1]])
        result = SpotNMatch(reference).match(target)
        results[label] = result[5]
        vresults[label] = result[3]

    p = ttest_ind(results["match"], results["mismatch"], alternative="greater")
    vp = ttest_ind(vresults["match"], vresults["mismatch"], alternative="greater")
    print(p, vp)

    fig = plot.figure(figsize=(4, 3), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    plt = fig.add_subplot()
    plt.boxplot(results.values(), whis=1.5, showfliers=True, labels=results.keys(), vert=False)
    # plot.xlim(0, 1)
    plot.yticks(rotation=90, va="center")
    # plt.set_xlabel("mismatch")
    plt.set_xlabel("fraction of residuals explained")
    # plt.set_title("filter-genes = none (HCA dataset)")
    fig.show()
    fig.savefig("filter_none.pdf")

    fig = plot.figure(figsize=(4, 3), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.15)
    plt = fig.add_subplot()
    plt.boxplot(vresults.values(), whis=1.5, showfliers=True, labels=vresults.keys(), vert=False)
    # plot.xlim(0, 1)
    plot.yticks(rotation=90, va="center")
    # plt.set_xlabel("mismatch")
    plt.set_xlabel("fraction of residuals explained")
    # plt.set_title("filter-genes = evaluate (HCA dataset)")
    fig.savefig("filter_evaluate.pdf")


if __name__ == "__main__":
    generate_data()
    generate_table(ROWS, COLUMNS)
