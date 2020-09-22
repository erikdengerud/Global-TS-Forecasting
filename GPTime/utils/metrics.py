"""
Forecasting metrics.
Functions scoring predictions.
"""
import numpy as np
import sys
import logging

sys.path.append("")
from GPTime.config import cfg

logger = logging.getLogger(__name__)


def MASE(actual: np.array, predicted: np.array, scale: np.array) -> float:
    """ Calculating Mean Absolute Scaled Error """
    return np.mean(np.nanmean(np.abs(actual - predicted), axis=1) / scale)


def SMAPE(actual: np.array, predicted: np.array) -> float:
    """ Calculating Symmetric Mean Absolute Prediction Error """
    return 200.0 * np.nanmean(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)), axis=(1,0))


def OWA(mase: float, smape: float) -> float:
    """ Calculating the Overall Weighted Average used in M4 """
    return (
        mase / cfg.scoring.owa.naive2_mase + smape / cfg.scoring.owa.naive2_smape
    ) / 2


# Below TODO
def mae(actual, predicted):
    return 0


def rmse(actual, predicted):
    return 0


def mape(actual, predicted):
    return 0


def wape(actual, predicted):
    return 0