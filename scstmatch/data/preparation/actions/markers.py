from .action import Action


class FindMarkers(Action):
    def __init__(self, **kwargs):
        super().__init__()

    def perform(self, dataset_group):
        return dataset_group
