import argparse
import pathlib

from scstmatch.generation import CellTypeGenerator

NUMBER_OF_GENES = 500
NUMBER_OF_CELL_TYPES = 10
NUMBER_OF_BASELINE_GENES = 200
BASELINE_GENE_RANGE = (500, 100)
BASELINE_GENE_STD_RANGE = (500, 100)
NUMBER_OF_MARKER_GENES = 20
MARKER_GENE_RANGE = (500, 100)
MARKER_GENE_STD_RANGE = (50, 10)


def main():
    parser = argparse.ArgumentParser(description="Generate cell type specifications and export them.")
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-t", "--types", dest="n_types", help="The number of cell_types to generate",
                        type=int, default=NUMBER_OF_CELL_TYPES)
    parser.add_argument("-g", "--genes", dest="n_genes", help="The number of genes to generate",
                        type=int, default=NUMBER_OF_GENES)
    parser.add_argument("-m", "--markers", dest="n_marker", help="The number of marker genes per cell type",
                        type=int, default=NUMBER_OF_MARKER_GENES)
    parser.add_argument("-b", "--baseline", dest="n_baseline",
                        help="The number of baseline genes across all cell types",
                        type=int, default=NUMBER_OF_BASELINE_GENES)

    args = parser.parse_args()
    g = CellTypeGenerator(n_genes=args.n_genes, n_types=args.n_types, n_marker=args.n_marker,
                          n_baseline=args.n_baseline)
    g.generate().write(args.out_file)


if __name__ == "__main__":
    main()
