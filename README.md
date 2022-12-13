
# Spatial Transcriptomics Deconvolution Data Matching

This repository contains the data generation, deconvolution, and data matching tools for spatial transcriptomics described in my Master's thesis.


Here is a short example of how to use these tools.
```python
from scstmatch.data import SingleCellDataset, SpatialTranscriptomicsDataset
from scstmatch.generation import SC2STGenerator
from scstmatch.deconvolution import IntegralDeconvolver, GreedySelector
from scstmatch.deconvolution.evaluation import evaluate_jsd
from scstmatch.matching import SpotNMatch

sc = SingleCellDataset.read("single-cell-data.h5ad")
sc.cell_type_column = "CELLTYPE"

real_st = SpatialTranscriptomicsDataset.read("spatial-data.h5ad")
synthetic_sc = SC2STGenerator(sc).generate()

type_mixtures = IntegralDeconvolver(sc, GreedySelector()).deconvolve(synthetic_sc)
spot_jsd = evaluate_jsd(synthetic_sc, type_mixtures)

scores

```


## Setup

To install the dependencies, setup a virtual environment and install the `requirements.txt`:
```bash
python -m venv .venv
source .venv/bin/active
pip install -r requirements.txt
```

To download the source dataset ([HCA](https://www.heartcellatlas.org)) and generate the variants, use `data/formula.py`:
```bash
cd data
# This might take a while.
python formula.py
cd ..
```

## Project structure

The primary libraries for single-cell and spatial transcriptomics data handling are located in `scstmatch`.
These are structured into four main components:
* `data` handles dataset management and utility functions
* `generation` handles fully and partially synthetic dataset generation functions
* `deconvolution` implements the integral deconvolution approach described in the thesis
* `matching` implements the SpotNMatch Algorithm
