from scstmatch.data.preparation.formula import DatasetFormula

# TODO: Investigate using make for this instead, for automatic parallelism, or even a workload manager.
formula = DatasetFormula("datasets.yml")
formula.setup()
