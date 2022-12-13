import argparse
import pathlib

from scstmatch.data import CellTypeDataset
from scstmatch.generation import SingleCellGenerator

NUMBER_OF_CELLS = 10000
COUNTS_PER_CELL = 1000


def main():
    parser = argparse.ArgumentParser(description="Generate a single cell dataset from a cell type specification.")
    parser.add_argument("-i", "--in", dest="in_file", help="The path to the input cell type specification",
                        required=True, type=pathlib.Path)
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-s", "--samples", dest="n_samples", help="The number of (cell)samples to generate",
                        type=int, default=NUMBER_OF_CELLS)
    parser.add_argument("-c", "--counts", dest="n_counts", help="The total gene-count per cell",
                        type=int, default=COUNTS_PER_CELL)

    args = parser.parse_args()
    generator = SingleCellGenerator(
        cell_spec=CellTypeDataset.read(args.in_file),
        n_samples=args.n_samples, n_counts=args.n_counts)
    generator.generate().write(args.out_file)


if __name__ == "__main__":
    main()
