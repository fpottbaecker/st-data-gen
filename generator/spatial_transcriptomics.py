import argparse
import pathlib

from scstmatch.data import CellTypeDataset
from scstmatch.generation import SpatialTranscriptomicsGenerator

NUMBER_OF_SPOTS = 1000
COUNTS_PER_SPOT = 1000
CELLS_PER_SPOT = 10


def main():
    parser = argparse.ArgumentParser(description="Generate a spatial transcriptomics dataset from a cell type specification.")
    parser.add_argument("-i", "--in", dest="in_file", help="The path to the input cell type specification",
                        required=True, type=pathlib.Path)
    parser.add_argument("-o", "--out", dest="out_file", help="The output file",
                        required=True, type=pathlib.Path)
    parser.add_argument("-s", "--spots", dest="n_spots", help="The number of spots to generate",
                        type=int, default=NUMBER_OF_SPOTS)
    parser.add_argument("-c", "--counts", dest="n_counts", help="The total gene-count per spot",
                        type=int, default=COUNTS_PER_SPOT)
    parser.add_argument("-n", "--cells", dest="n_cells", help="The number of cells per spot",
                        type=int, default=CELLS_PER_SPOT)

    args = parser.parse_args()
    generator = SpatialTranscriptomicsGenerator(
        cell_spec=CellTypeDataset.read(args.in_file),
        n_spots=args.n_spots, n_counts=args.n_counts, n_cells=args.n_cells)
    generator.generate().write(args.out_file)


if __name__ == "__main__":
    main()
