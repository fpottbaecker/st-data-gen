import numpy as np
import pandas as pd
import scipy as sp

from scstmatch.data import SpatialTranscriptomicsDataset


def match_results(actual: pd.DataFrame, predicted: pd.DataFrame):
    """
    Expand the two data frames to include all columns. Note that this requires both data frames to have the same index.
    :param actual: The known cell type mixture data frame
    :param predicted: The predicted cell type mixture data frame
    :return: both datasets (actual, predicted) expanded to the union of columns (cell types), missing values are set to 0.
    """
    actual_columns = actual.columns
    predicted_columns = predicted.columns

    full_actual = actual.copy(deep=True)
    full_predicted = predicted.copy(deep=True)
    full_actual[predicted_columns.difference(actual_columns)] = 0
    full_predicted[actual_columns.difference(predicted_columns)] = 0

    return full_actual, full_predicted[full_actual.columns]


def evaluate_jsd(target: SpatialTranscriptomicsDataset, predicted: pd.DataFrame):
    """
    :param target: A target dataset with a set "Y" obsm containing the known cell type mixtures
    :param predicted: The predicted data frame
    :return: The JSD per prediction spot
    """
    actual, predicted = match_results(target.anndata.obsm["Y"], predicted)
    dists = sp.spatial.distance.jensenshannon(actual, predicted, axis=1)
    dists[np.isnan(dists)] = 0
    return dists


def evaluate_rmse(target: SpatialTranscriptomicsDataset, predicted: pd.DataFrame):
    """
    :param target: A target dataset with a set "Y" obsm containing the known cell type mixtures
    :param predicted: The predicted data frame
    :return: The RMSE per prediction spot
    """
    actual, predicted = match_results(target.anndata.obsm["Y"], predicted)
    squared_errors = (actual - predicted) ** 2
    return np.sqrt(np.mean(squared_errors, axis=1))
