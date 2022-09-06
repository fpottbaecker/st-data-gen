
from .actions import Action, actions


class DatasetVariant:
    steps: list[Action]

    def __init__(self, formula, /, source, steps, **kwargs):
        self.source = formula.single_cell_sources[source]
        self.steps = [actions[data["action"]](**data) for data in steps]

    def setup(self):
        dataset = self.source.setup()
        for step in self.steps:
            dataset = step.perform(dataset)
        return dataset
