import itertools
import multiprocessing
import anndata as ad
from functools import partial
import matplotlib.pyplot as plot

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.matching import SPOTLightMatcher

FILE_A = "../data/reference/hca_sanger_gender_Female.sc.h5ad"
FILE_B = "../data/reference/hca_harvard_gender_Male_-muscles.sc.h5ad"

COLUMNS = {
    ("Harvard", None, "F"): "../data/reference/hca_harvard_gender_Female.sc.h5ad",
    ("Harvard", None, "M"): "../data/reference/hca_harvard_gender_Male.sc.h5ad",
    ("Harvard", "V1", "F"): "../data/reference/hca_harvard_gender_Female_-muscles.sc.h5ad",
    ("Harvard", "V1", "M"): "../data/reference/hca_harvard_gender_Male_-muscles.sc.h5ad",
    ("Harvard", "V2", "F"): "../data/reference/hca_harvard_gender_Female_-endothelial.sc.h5ad",
    ("Harvard", "V2", "M"): "../data/reference/hca_harvard_gender_Male_-endothelial.sc.h5ad",
    ("Sanger", None, "F"): "../data/reference/hca_sanger_gender_Female.sc.h5ad",
    ("Sanger", None, "M"): "../data/reference/hca_sanger_gender_Male.sc.h5ad",
    ("Sanger", "V1", "M"): "../data/reference/hca_sanger_gender_Male_-muscles.sc.h5ad",
}

ROWS = {
    ("Harvard", None, "F"): "../data/st/hca_harvard_donor_H6.st.h5ad",
    ("Harvard", None, "M"): "../data/st/hca_harvard_donor_H3.st.h5ad",
    ("Harvard", "V1", "F"): "../data/st/hca_harvard_donor_H6_-muscles.st.h5ad",
    ("Harvard", "V1", "M"): "../data/st/hca_harvard_donor_H3_-muscles.st.h5ad",
    ("Harvard", "V2", "F"): "../data/st/hca_harvard_donor_H6_-endothelial.st.h5ad",
    ("Harvard", "V2", "M"): "../data/st/hca_harvard_donor_H3_-endothelial.st.h5ad",
    ("Sanger", None, "F"): "../data/st/hca_sanger_donor_D5.st.h5ad",
    ("Sanger", None, "M"): "../data/st/hca_sanger_donor_D2.st.h5ad",
}


def generate_cell(column, row):
    column, column_path = column
    row, row_path = row
    print(f"BEGIN {column}, {row}")
    reference = SingleCellDataset.read(column_path)
    target = SpatialTranscriptomicsDataset.read(row_path)
    result = SPOTLightMatcher().match(reference, target)
    print(f"END {column}, {row}")
    return column, row, result


def generate_column(rows, head):
    header, head_path = head
    print(f"BEGIN {header}")
    reference = SingleCellDataset.read(head_path)
    matcher = SPOTLightMatcher()
    result = []
    for _, path in rows.items():
        target = SpatialTranscriptomicsDataset.read(path)
        result.append(matcher.match(reference, target))
        del target
    print(f"END {header}")
    del reference
    return result


def identify_best(data, column, element):
    column = data[column]
    if element != column.max():
        return f"{element:0.3f}"
    else:
        return f"\\emph{{{element:0.3f}}}"


def generate_table(rows, columns):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        data = np.array(p.map(partial(generate_column, rows), columns.items()))

        df_evaluate = pd.DataFrame(data[:, :, 0], columns=rows.keys(), index=columns.keys())
        df_fit = pd.DataFrame(data[:, :, 1], columns=rows.keys(), index=columns.keys())
        df = pd.DataFrame(data[:, :, 2], columns=rows.keys(), index=columns.keys())
        # data = p.starmap(generate_cell, itertools.product(columns.items(), rows.items()))
        df.index.name = "Target"
        df.columns.name = "Reference"

    #formatters = map(lambda x: partial(identify_best, df, x), range(0, df.shape[1]))

    # .style.highlight_max(color="black", props="bfseries:").set_precision(4).to_latex(hrules=True, clines="skip-last;index")
    print(df.style.highlight_max(color="black", props="bfseries:").set_precision(4).to_latex())


def generate_data():
    pairs = {
        "match": (("Harvard", None, "F"), ("Harvard", None, "F")),
        "mismatch": (("Sanger", "V1", "M"), ("Harvard", None, "F")),
        #"none/V1": ("F(h) -muscles", "H6(f)"),
        #"none/V2": ("F(h) -endothelial", "H6(f)"),
        #"gender/V1": ("M(h) -muscles", "H6(f)"),
        #"gender/V2": ("M(h) -endothelial", "H6(f)"),
    }

    results = {}
    vresults = {}

    plot.rcParams.update({
        "text.usetex": True,
    })

    for label, paired in pairs.items():
        reference = SingleCellDataset.read(COLUMNS[paired[0]])
        target = SpatialTranscriptomicsDataset.read(ROWS[paired[1]])
        result = SPOTLightMatcher().match(reference, target)
        results[label] = result[5]
        vresults[label] = result[3]

    p = ttest_ind(results["match"], results["mismatch"], alternative="greater")
    vp = ttest_ind(vresults["match"], vresults["mismatch"], alternative="greater")
    print(p, vp)

    fig = plot.figure(figsize=(8, 3), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15)
    plt = fig.add_subplot()
    plt.boxplot(results.values(), whis=1.5, showfliers=True, labels=results.keys(), vert=False)
    plot.xlim(0, 1)
    plot.yticks(rotation=90, va="center")
    #plt.set_xlabel("mismatch")
    plt.set_xlabel("fraction of residuals explained")
    #plt.set_title("filter-genes = none (HCA dataset)")
    fig.show()
    fig.savefig("filter_none.pdf")

    fig = plot.figure(figsize=(8, 3), dpi=300)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15)
    plt = fig.add_subplot()
    plt.boxplot(vresults.values(), whis=1.5, showfliers=True, labels=vresults.keys(), vert=False)
    plot.xlim(0, 1)
    plot.yticks(rotation=90, va="center")
    # plt.set_xlabel("mismatch")
    plt.set_xlabel("fraction of residuals explained")
    #plt.set_title("filter-genes = evaluate (HCA dataset)")
    fig.savefig("filter_evaluate.pdf")


if __name__ == "__main__":
    generate_data()
    #generate_table(ROWS, COLUMNS)
