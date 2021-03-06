"""
Calculating the overall M4 score for a model
"""
import torch.nn as nn
import torch
import importlib
import pandas as pd
import glob
import logging
import numpy as np
from typing import Dict, List, Tuple
import sys

sys.path.append("")

logger = logging.getLogger(__name__)

from GPTime.config import cfg
from GPTime.utils.metrics import MASE, SMAPE, OWA
from GPTime.utils.scaling import MASEScaler

Scaler = getattr(
    importlib.import_module(cfg.train.scaler_module), cfg.train.scaler_name
)


def period_from_fname(fname: str, period_dict: Dict) -> Tuple[int, str]:
    """
    Get the periodicity from a filename as integer and string.
    """
    for p in period_dict.keys():
        if p.lower() in fname.lower():
            return period_dict[p], p
    logger.warning(f"No season or period found for file {fname}.")
    return 1, ""


def create_training_data(fname: str, memory: int) -> np.array:
    """
    Reverting the training data.
    The data is in a form where the first value is in the first column etc. and it is
    padded at the end with nans to get the same length for all ts. This function changes
    the format s.t. the last column is the first value etc.
    """
    Y = pd.read_csv(fname, index_col=0).to_numpy()
    X = np.zeros((Y.shape[0], memory))
    X_mask = np.zeros((Y.shape[0], memory))
    for i in range(Y.shape[0]):
        ts = Y[i][~np.isnan(Y[i])]
        ts = ts[-memory:]
        X[i, -len(ts):] = ts
        X_mask[i, -len(ts):] = 1.0
    return X, X_mask


def multi_step_predict(
    model: nn.Module, train_data: np.array, mask_data: np.array, horizon: int, frequencies: np.array, period_str:str=None,  encode_frequencies:bool=False,
) -> np.array:
    """
    Multi step forecasting with a model on training data.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    memory = train_data.shape[1]
    #logger.debug(f"frequencies.shape: {frequencies.shape}")
    if np.max(frequencies) > memory:
        frequencies = np.array([1 for _ in range(train_data.shape[0])])
    freq_str_arr = np.expand_dims(np.repeat(period_str, train_data.shape[0]), axis=1)
    with torch.no_grad():
        for i in range(horizon):
            sample = torch.from_numpy(train_data[:, -memory:]).to(device)
            sample_mask = torch.from_numpy(mask_data[:,-memory:]).to(device)
            last_period = torch.from_numpy(sample.shape[1] - frequencies).to(device)
            #logger.debug(f"last_period.shape: {last_period.shape}")
            #logger.debug(f"last_period[0]: {last_period[0]}")
            #logger.debug(last_period.shape)
            #logger.debug(last_period[0])
            out = model(sample, sample_mask, last_period, freq_str_arr).cpu().detach().numpy()
            train_data = np.hstack((train_data, out))
            mask = np.hstack((mask_data, np.ones((mask_data.shape[0], 1))))
    forecast = train_data[:, -horizon:]
    return forecast


def predict_M4(model: nn.Module, scale: bool=False, seasonal_init:bool=False, val_set:bool=False, freq:str=None, encode_frequencies:bool=False) -> np.array:
    """ Predicting M4 using a model provided. """
    assert hasattr(model, "forward")
    #assert hasattr(model, "memory")
    model.eval()

    if val_set:
        all_train_files = glob.glob(cfg.path.m4_val_train + "*")
    else:
        all_train_files = glob.glob(cfg.path.m4_train + "*")
    all_train_files.sort()
    
    #logger.info(f"freq is {freq}")
    if freq is not None:
        #logger.info(f"Keeping requrency {freq}")
        keep_files = []
        for fname in all_train_files:
            _, period_str = period_from_fname(fname=fname, period_dict=cfg.scoring.m4.periods)
            if period_str[0].lower() == freq.lower():
                keep_files.append(fname)
        all_train_files = keep_files
    #logger.info(all_train_files)
    assert len(all_train_files) > 0, f"Did not find data in {cfg.path.m4_train}"
    frames = []
    for fname in all_train_files:
        period_numeric, period_str = period_from_fname(
            fname=fname, period_dict=cfg.scoring.m4.periods
        )

        if hasattr(model, "memory"):
            memory = getattr(model, "memory")
        else:
            memory = cfg.train.model_params_mlp.in_features
        X, X_mask = create_training_data(fname=fname, memory=memory)

        if scale:
            max_scale = np.expand_dims(np.max(X, 1), axis=1)                                                                                         
            X = np.divide(X, max_scale)

        if seasonal_init:
            frequencies = np.array([period_numeric for _ in range(X.shape[0])])
        else:
            frequencies = np.array([1 for _ in range(X.shape[0])])

        predictions = multi_step_predict(
            model=model,
            train_data=X,
            mask_data=X_mask,
            horizon=cfg.scoring.m4.horizons[period_str],
            frequencies=frequencies,
            period_str=period_str,
            encode_frequencies=encode_frequencies,
        )

        if scale:
            predictions = np.multiply(predictions, max_scale)

        df = pd.DataFrame(predictions)
        #df.index = [f"{period_str[0].upper()}{i}" for i in range(1, len(df)+1)]
        df["id"] = [f"{period_str[0].upper()}{i}" for i in range(1, len(df)+1)]
        #df.index.name = "id"
        frames.append(df)
        
    #logger.info("len of list of dfs: {len(frames)}")
    df_all = pd.concat(frames, sort=False)
    df_all = df_all.set_index("id")
    #logger.info(df_all.head())
    df_all.columns = [f"V{i}" for i in range(1, len(df_all.columns)+1)]
    predictions = df_all.values
    #logger.info("evaluate: shape: {df.shape}")
    return df_all.values, df_all


def score_M4(
    predictions: np.array, df_results_name: str = "GPTime/results/M4/test.csv", val:bool=False
) -> Dict:
    """ Calculating the OWA. Return dict of scores of subfrequencies also."""
    frequency_metrics: Dict[str, Dict[str, float]] = {}
    # Read in and prepare the data
    if val:
        all_test_files = glob.glob(cfg.path.m4_val_test + "*")
        all_train_files = glob.glob(cfg.path.m4_val_train + "*")
    else:
        all_test_files = glob.glob(cfg.path.m4_test + "*")
        all_train_files = glob.glob(cfg.path.m4_train + "*")
    # Removing hourly for the zero-shot part
    #all_train_files = [fname for fname in all_train_files if "hourly" not in fname.lower()]
    #all_test_files = [fname for fname in all_test_files if "hourly" not in fname.lower()]
    all_test_files.sort()
    all_train_files.sort()
    crt_pred_index = 0
    tot_mase = 0.0
    tot_smape = 0.0
    for fname_train, fname_test in zip(all_train_files, all_test_files):
        #logger.info(fname_test)
        #logger.info(fname_train)
        df_train = pd.read_csv(fname_train, index_col=0)
        df_test = pd.read_csv(fname_test, index_col=0)
        period_num, period_str = period_from_fname(
            fname=fname_train, period_dict=cfg.scoring.m4.periods
        )
        horizon = cfg.scoring.m4.horizons[period_str]

        Y = df_test.values[:, :horizon]
        index = crt_pred_index + Y.shape[0]
        predicted = predictions[crt_pred_index:index, :horizon]
        #logger.info(f"predicted.shape: {predicted.shape}")
        
        assert np.sum(np.isnan(Y)) == 0, "NaNs in Y"
        assert np.sum(np.isnan(predicted)) == 0, f"NaNs in predictions: {np.where(np.isnan(predicted))}"
        assert Y.shape == predicted.shape, "Y and predicted have different shapes"

        #scale = Scaler().fit(df_train.values, freq=period_num).scale_.flatten()
        scale = MASEScaler().fit(df_train.values, freq=period_num).scale_.flatten()

        mase_freq = MASE(Y, predicted, scale)
        smape_freq = SMAPE(Y, predicted)
        owa_freq = OWA(mase=mase_freq, smape=smape_freq, freq=period_str)
        tot_mase += mase_freq * Y.shape[0]
        tot_smape += smape_freq * Y.shape[0]
        #logger.debug(f"mase_freq = {mase_freq}")
        #logger.debug(f"smape_freq = {smape_freq}")
        frequency_metrics[period_str] = {}
        frequency_metrics[period_str]["MASE"] = mase_freq
        frequency_metrics[period_str]["SMAPE"] = smape_freq
        frequency_metrics[period_str]["OWA"] = owa_freq

        crt_pred_index += Y.shape[0]
    
    tot_mase = tot_mase / crt_pred_index
    tot_smape = tot_smape / crt_pred_index
    tot_owa = OWA(tot_mase, tot_smape, freq="global")

    frequency_metrics["GLOBAL"] = {}
    frequency_metrics["GLOBAL"]["MASE"] = tot_mase
    frequency_metrics["GLOBAL"]["SMAPE"] = tot_smape
    frequency_metrics["GLOBAL"]["OWA"] = tot_owa

    df = pd.DataFrame(frequency_metrics).T
    df.to_csv(df_results_name)

    return frequency_metrics
