import importlib
import logging
import os
import torch
import sys
import glob
import pandas as pd

sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scoring import predict_M4, score_M4

logger = logging.getLogger(__name__)

def evaluate(evaluate_cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = getattr(importlib.import_module(evaluate_cfg.model_module), evaluate_cfg.model_name)
    # load model
    if Model.__name__ == "MLP":
        model_params = evaluate_cfg.model_params_mlp
    elif Model.__name__ == "AR":
        model_params = evaluate_cfg.model_params_ar
    elif Model.__name__ == "TCN":
        model_params = evaluate_cfg.model_params_tcn
    else:
        logger.warning("Unknown model name.")
    if evaluate_cfg.global_model:
        logger.info("Evaluating global model")
        model = Model(**model_params).double()
        model_path = os.path.join(evaluate_cfg.model_save_path, evaluate_cfg.name + ".pt")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        preds, df_preds = predict_M4(model=model, scale=evaluate_cfg.scale, seasonal_init=evaluate_cfg.seasonal_init, val_set=evaluate_cfg.val_set, encode_frequencies=evaluate_cfg.model_params_mlp.encode_frequencies)
        result_file = os.path.join(evaluate_cfg.result_path, "result.csv")
        logger.info(f"results fiel: {result_file}")
        d = score_M4(preds, df_results_name=result_file, val=evaluate_cfg.val_set)
        logger.info(d)
        csv_path = os.path.join(evaluate_cfg.predictions_path, "forecast.csv")
        df_preds.to_csv(csv_path)
    else:
        horizons = {
            "Y": 6,
            "Q": 8,
            "M": 18,
            "W": 13,
            "D": 14,
            "H": 48,
            }
        # find all models
        all_model_paths = glob.glob(os.path.join(evaluate_cfg.model_save_path, "*.pt"))
        all_model_paths.sort()
        logger.info(all_model_paths)
        all_dfs = []
        for model_path in all_model_paths:
            logger.info(f"predicting model: {model_path}")
            logger.info(model_params)
            model_params["in_features"] = evaluate_cfg.lookback * horizons[model_path[-4]]
            model = Model(**model_params).double()
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
            logger.info(model_path)
            logger.info(model_path[0])
            preds, df_preds = predict_M4(model=model, scale=evaluate_cfg.scale, seasonal_init=evaluate_cfg.seasonal_init, val_set=evaluate_cfg.val_set, freq=model_path[-4])
            all_dfs.append(df_preds)
            logger.info(df_preds.head())
        # concat dataframes
        df_all = pd.concat(all_dfs, sort=False)
        preds = df_all.values
        # save etc
        result_file = os.path.join(evaluate_cfg.model_save_path, "result.csv")
        d = score_M4(preds, df_results_name=result_file, val=evaluate_cfg.val_set)
        logger.info(d)
        csv_path = os.path.join(evaluate_cfg.model_save_path, "forecast.csv")
        df_all.to_csv(csv_path)
