import logging
import numpy as np
from typing import Tuple
import sys

sys.path.append("")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from GPTime.config import cfg

logger = logging.getLogger(__name__)


def MASEscale(ts: np.array, freq: str) -> float:  # Tuple[np.array, float]:
    """ Scale time series using scaling from MASE """
    d = {
        "Y": "yearly",
        "Q": "quarterly",
        "M": "monthly",
        "W": "weekly",
        "D": "daily",
        "H": "hourly",
        "O": "other",
    }
    period = cfg.scoring.m4.periods[d[freq]]
    if len(ts) <= period:
        period = 1
    scale = np.mean(np.abs((ts - np.roll(ts, shift=period))[period:]))
    if scale == 0:
        scale = ts[0] if ts[0] != 0 else 1
    assert scale > 0, f"Scale is not positive! {ts}, scale: {scale}"
    return scale


class MASEScaler(TransformerMixin, BaseEstimator):
    """
    Scaler as in sklearn.preprocessing.
    Replacing 0s, infs and nans with 1.
    """

    def __init__(self, seasonality: bool = True):
        self.seasonality = seasonality

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "scale_"):
            del self.scale_

    def fit(self, X: np.array, freq: int, axis=1) -> object:
        self._reset()
        return self.partial_fit(X, freq, axis=axis)

    def partial_fit(self, X: np.array, freq: int, axis: int = 1) -> object:
        #logger.debug(f"freq = {freq}")
        #logger.debug(f"type(freq)={type(freq)}")

        try:
            if len(X.shape) == 1:
                if len(X) <= freq:
                    freq = 1
                scale = np.mean(np.abs(X-np.roll(X, shift=freq))[freq:])
                scale = 1 if scale == 0 else scale
                scale = 1 if np.isinf(scale) else scale
                scale = 1 if np.isnan(scale) else scale
                self.scale_ = scale
            elif len(X.shape) == 2:
                if X.shape[1] <= freq:
                    freq = 1
                scale = np.nanmean(
                    np.abs(X - np.roll(X, shift=freq, axis=axis))[:, freq:], axis=axis
                )
                scale[scale == 0] = 1
                scale[np.isinf(scale)] = 1
                scale[np.isnan(scale)] = 1
                self.scale_ = np.expand_dims(scale, axis=axis)
            else:
                raise Exception("Input array not of valid shape. Must be one or two-dimensional.")
        except Exception as e:
            logger.warning(e)
            self.scale_ = None
        # logger.info(self.scale_)
        return self

    def transform(self, X: np.array) -> np.array:
        """
        Scale the data.
        """
        check_is_fitted(self)
        # logger.info(X.shape)
        # logger.info(self.scale_.shape)
        #logger.debug(f"X.dtype : {X.dtype}")
        #logger.debug(f"self.scale_.dtype : {self.scale_.dtype}")

        X = X.astype(float)
        X /= self.scale_
        return X

    def inverse_transform(self, X: np.array) -> np.array:
        """
        Scale back the data to original scale.
        """
        check_is_fitted(self)
        X *= self.scale_
        return X

class NoScaler(TransformerMixin, BaseEstimator):
    """Scaler that does no scaling.

    """
    def __init__(self, copy=True):
        super(NoScaler, self).__init__()

    def fit(self, X: np.array) -> object:
        """Fits the scaler to the dataset by doing nothing.

        Args:
            X (np.array): Data matrix, (num_samples, num_observations)

        Returns:
            object: self
        """
        return self

    def transform(self, X: np.array, **kwargs) -> np.array:
        """Transforming a data matrix.

        Args:
            X (np.array): Data matrix, (num_samples, num_observations)

        Returns:
            np.array: Transformed data
        """
        return X

    def inverse_transform(self, X: np.array) -> np.array:
        """Transform back to the original scale.

        Args:
            X (np.array): Data matrix, (num_samples, num_observations)

        Returns:
            np.array: Inverse transformed data
        """
        return X
