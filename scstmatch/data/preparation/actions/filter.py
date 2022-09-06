from typing import Union

import numpy as np

from scstmatch.data import Dataset, DatasetGroup
from .action import Action


KEYS = ["obs", "var", "-obs", "-var"]


class FilterSpec:
    obs_include: dict[str, Union[list[any], any]]
    obs_exclude: dict[str, Union[list[any], any]]
    var_include: dict[str, Union[list[any], any]]
    var_exclude: dict[str, Union[list[any], any]]

    def __init__(self, spec):
        self.obs_include = spec.get("obs", {})
        self.obs_exclude = spec.get("-obs", {})
        self.var_include = spec.get("var", {})
        self.var_exclude = spec.get("-var", {})

    def filter(self, dataset: Dataset):
        ad = dataset.anndata
        obs_positive = [np.isin(ad.obs[key], value) for key, value in self.obs_include.items()]
        obs_negative = [np.isin(ad.obs[key], value, invert=True) for key, value in self.obs_exclude.items()]
        obs_mask = np.all([*obs_positive, *obs_negative, np.full(shape=ad.n_obs, fill_value=True)], axis=0)

        var_positive = [np.isin(ad.var[key], value) for key, value in self.var_include.items()]
        var_negative = [np.isin(ad.var[key], value, invert=True) for key, value in self.var_exclude.items()]
        var_mask = np.all([*var_positive, *var_negative, np.full(shape=ad.n_vars, fill_value=True)], axis=0)
        return dataset.copy_with(ad[obs_mask, var_mask])


class FilterAction(Action):
    filters: dict[str, FilterSpec]

    def __init__(self, /, variants: dict[str, dict[str, any]] = None, include_original=False, **kwargs):
        super().__init__(**kwargs)
        _single = any(key in kwargs for key in KEYS)
        _multiple = variants is not None
        if _single and _multiple:
            raise ValueError("Cannot simultaneously filter on a single conditions and multiple variants")
        if _single and include_original:
            raise ValueError("Cannot include original with single filter (use variants)")
        if _single:
            self.filters = {"": FilterSpec(kwargs)}
        else:
            self.filters = {key: FilterSpec(spec) for key, spec in variants.items()}
            if include_original:
                self.filters[""] = FilterSpec({})

    def perform(self, dataset_group: DatasetGroup):
        return dataset_group.split(lambda parent_key, dataset: {DatasetGroup.join_keys(parent_key, key): filter_spec.filter(dataset) for key, filter_spec in self.filters.items()})

