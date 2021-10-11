import numpy as np


def generate_expression_profile(cell_spec, n_genes, cell_type, rng: np.random.Generator):
    gene_p = np.ndarray(shape=n_genes, dtype="float64")
    weights = cell_spec.X[cell_type]
    weights_std = cell_spec.layers["std"][cell_type]
    gene_p = np.maximum(rng.normal(weights, weights_std), 0)
    gene_p /= gene_p.sum()
    return gene_p


def select_cells(n_cells, n_types, temperature: float, rng: np.random.Generator):
    # Assign temperatures
    p = np.full(n_cells, fill_value=n_types, dtype="float32") ** (np.arange(n_types, dtype="float32") * temperature)
    rng.shuffle(p)
    return rng.choice(n_types, n_cells, p=p/p.sum(), replace=True)
