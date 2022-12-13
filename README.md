
# Spatial Transcriptomics Deconvolution Data Matching

This repository contains the data generation, deconvolution, and data matching tools for spatial transcriptomics described in my Master's thesis.


Here is a short example of how to use these tools.
```python
from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.generation import SC2STGenerator
from scstmatch.deconvolution import IntegralDeconvolver, GreedySelector
from scstmatch.deconvolution.evaluation import evaluate_jsd
from scstmatch.matching import SpotNMatch

# Load a SC dataset and set the known cell type column
sc = SingleCellDataset.read("single-cell-data.h5ad")
sc.cell_type_column = "CELLTYPE"

# Load a real ST dataset and generate a synthetic one
real_st = SpatialTranscriptomicsDataset.read("spatial-data.h5ad")
synthetic_sc = SC2STGenerator(sc).generate()

# Deconvolve using the greedy selector and evaluate the JSD
type_mixtures = IntegralDeconvolver(sc, GreedySelector()).deconvolve(synthetic_sc)
spot_jsd = evaluate_jsd(synthetic_sc, type_mixtures)

# Get per spot matching scores
scores = SpotNMatch(sc).match(real_st)
```


## Setup

To install the dependencies, setup a virtual environment and install the `requirements.txt`:
```bash
python -m venv .venv
source .venv/bin/active
pip install -r requirements.txt
```

Depending on your setup, you might need to add the project directory to your `PYTHONPATH`.

To download the source dataset ([HCA](https://www.heartcellatlas.org)) and generate the variants, use `data/formula.py`:
```bash
cd data
# This might take a while.
python formula.py
cd ..
```

## Project Structure

### SCSTMatch Library

The primary libraries for single-cell and spatial transcriptomics data handling are located in `scstmatch`.
These are structured into four main components:
* `data` handles dataset management and utility functions
* `generation` handles fully and partially synthetic dataset generation functions
* `deconvolution` implements the integral deconvolution approach described in the thesis
* `matching` implements the SpotNMatch Algorithm


### Data

The `data` folder contains the definitions for the reference datasets and a script to generate them.
This is used by different scripts for the evaluation of this thesis.

### Thesis Content

The `thesis` folder contains notebooks and scripts to generate figures and tables for the thesis.
This includes [AntiSplodge](https://github.com/HealthML/AntiSplodge) training and score/deconvolution correlation.

### Scripts

The `scripts` folder contains utility scripts and older evaluation methods.
This also includes CLI frontends of the data generators: `cell_types.py`, `single_cell.py`, `spatial_transcriptomics.py`, `st_from_sc.py`.
