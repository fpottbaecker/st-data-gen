import argparse
import pathlib

from scstmatch.data import SingleCellDataset
from scstmatch.generation import SC2STGenerator

NUMBER_OF_SPOTS = 1000
CELLS_PER_SPOT = 10  # TODO: Maybe have a range here


def main():
    parser = argparse.ArgumentParser(
        description="Generate a spatial transcriptomics dataset from a single cell dataset.")
    parser.add_argument("-i", "--in", dest="in_file", help="The path to the input single cell dataset",
                        required=True, type=pathlib.Path)
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-s", "--spots", dest="n_spots", help="The number of spots to generate",
                        type=int, default=NUMBER_OF_SPOTS)
    parser.add_argument("-n", "--cells", dest="n_cells", help="The number of cells per spot",
                        type=int, default=CELLS_PER_SPOT)

    args = parser.parse_args()
    generator = SC2STGenerator(sc_data=SingleCellDataset.read(args.in_file),
                               n_spots=args.n_spots, n_cells=args.n_cells)
    generator.generate().write(args.out_file)


if __name__ == "__main__":
    main()
