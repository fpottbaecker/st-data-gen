import pandas as pd
import scipy as sp
import numpy as np

from scstmatch.data import SpatialTranscriptomicsDataset


def match_results(actual: pd.DataFrame, predicted: pd.DataFrame):
    actual_columns = actual.columns
    predicted_columns = predicted.columns

    full_actual = actual.copy(deep=True)
    full_predicted = predicted.copy(deep=True)
    full_actual[predicted_columns.difference(actual_columns)] = 0
    full_predicted[actual_columns.difference(predicted_columns)] = 0

    return full_actual, full_predicted[full_actual.columns]


def evaluate_jsd(target: SpatialTranscriptomicsDataset, predicted: pd.DataFrame):
    actual, predicted = match_results(target.anndata.obsm["Y"], predicted)

    return sp.spatial.distance.jensenshannon(actual, predicted, axis=1)


def evaluate_rmse(target: SpatialTranscriptomicsDataset, predicted: pd.DataFrame):
    actual, predicted = match_results(target.anndata.obsm["Y"], predicted)
    squared_errors = (actual - predicted)**2
    return np.sqrt(np.mean(squared_errors, axis=1))
