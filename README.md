
# Spatial Transcriptomics Deconvolution Data Matching

This repository contains the data generation, deconvolution, and data matching tools described in my Master's thesis.

## Setup

To install the dependencies, setup a virtual environment and install the `requirements.txt`:
```bash
python -m venv .venv
source .venv/bin/active
pip install -r requirements.txt
```

Run the separate notebooks in `notebooks/import` to download and set up the raw datasets.
<!-- TODO: run all -->

After downloading the datasets, you can run `data/formula.py` to set up the experimental synthetic data.

## Project structure

The primary libraries for single-cell and spatial transcriptomics data handling are located in `scstmatch`.
These are structured into four main components:
* `data` handles dataset management and utility functions
* `generation` handles fully and partially synthetic dataset generation functions
* `deconvolution` implements the integral deconvolution approach described in the thesis
* `matching` implements the SpotNMatch Algorithm
