import numpy as np


def generate_expression_profile(cell_spec, n_genes, cell_type, rng: np.random.Generator):
    gene_p = np.ndarray(shape=n_genes, dtype="float32")
    weights = cell_spec.X[cell_type]
    weights_std = cell_spec.layers["std"][cell_type]
    gene_p = np.maximum(rng.normal(weights, weights_std), 0)
    gene_p /= gene_p.sum()
    return gene_p
