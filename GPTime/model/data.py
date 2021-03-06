import torch
from torch.utils.data import Dataset

import logging
import glob
import json
import os
import numpy as np
import time
from typing import Dict, List, Tuple
import importlib
import sys

sys.path.append("")

from GPTime.config import cfg
from GPTime.utils.scaling import MASEScaler

Scaler = getattr(importlib.import_module(cfg.train.scaler_module), cfg.train.scaler_name)

logger = logging.getLogger(__name__)

class TSDataset(Dataset):
    """
    Time series dataset.
    """
    def __init__(
        self, 
        memory:int, 
        dataset_paths:Dict[str, str], 
        frequencies:Dict[str, bool],
        min_lengths:Dict[str, int],
        cutoff_date:str=None,  
        convolutions:bool=False,
        ds_type:str="full",
        lookback:int=4,
        ) -> None:
        super(TSDataset, self).__init__()
        #assert memory >= max(min_lengths.values()), "Increase model memory, it is shorter than max of min lengths."
        assert ds_type in ["train", "val", "full"]
        self.memory = memory
        self.convolutions = convolutions
        self.min_lengths = min_lengths
        self.cutoff_date = cutoff_date
        if cutoff_date is not None:
            self.cutoff_date_date = time.strptime(cutoff_date, "%Y-%m-%d")

        if ds_type == "val":
            self.cut_window_length = {
                "Y" : 6,
                "Q" : 8,
                "M" : 18,
                "W" : 13,
                "D" : 14,
                "H" : 48,
            }
        else:
            self.cut_window_length = {
                "Y" : 9,
                "Q" : 12,
                "M" : 27,
                "W" : 130,
                "D" : 140,
                "H" : 480,
            }
        self.lookback = lookback
        self.horizons = {
            "Y" : 6,
            "Q" : 8,
            "M" : 18,
            "W" : 13,
            "D" : 14,
            "H" : 48,
        }
        self.ds_type = ds_type

        # Read data into memory
        all_ts = []
        for dataset_path in dataset_paths.values():
            dirs = glob.glob(os.path.join(dataset_path, "*"))
            for d in dirs:
                logger.info(f"Loading dir: {d}")
                fnames = glob.glob(os.path.join(d, "*"))
                for fname in fnames:
                    with open(fname, "r") as fp:
                        ts_list = json.load(fp)
                        fp.close()
                    for ts in ts_list:
                        if frequencies[ts["frequency"]]:
                            prepared_ts = self.prepare_ts(ts)
                            if prepared_ts["values"].size:
                                all_ts.append(prepared_ts)
        self.all_ts = all_ts
        logger.info(f"Total of {len(all_ts)} time series in memory.")
        logger.info(f"Frequencies: {frequencies}.")

    def __len__(self):
        return len(self.all_ts)

    def __getitem__(self, idx) -> Tuple:
        id_ts = self.all_ts[idx]
        frequency = id_ts["frequency"]
        values = id_ts["values"]

        sample, label, mask = self.sample_ts_strict(values, frequency)
        label = label.reshape(-1,1).flatten() #to get correct shape

        sample_tensor = torch.from_numpy(sample)
        label_tensor = torch.from_numpy(label)
        sample_mask = torch.from_numpy(mask)
        label_mask = torch.from_numpy(np.ones(len(label)))

        if self.convolutions:
            sample_tensor = sample_tensor.unsqueeze(0)
            sample_mask = sample_mask.unsqueeze(0)
        #last_period = len(sample) - cfg.dataset.scaling.periods[frequency]
        freq_int = cfg.dataset.scaling.periods[frequency]

        return sample_tensor, label_tensor, sample_mask, label_mask, freq_int, frequency #last_period, freq_int
    
    def sample_ts_strict(self, ts:np.array, freq:str) -> Tuple[np.array, np.array, np.array]:
        """Sampling a training sample of a time series. Stricter in the sense that it samples
        cutoff points closer to the end of the training data.

        Args:
            ts (np.array): A time series in the form of an array.
            freq (str): The frequency of the time series. One capitalized letter.

        Returns:
            Tuple[np.array, np.array, np.array]: The sample, label and mask.
        """
        cut_window_len = self.cut_window_length[freq]

        sample = np.zeros(self.memory)
        sample_mask = np.zeros(self.memory)
        cut_point = np.random.randint(low=max(1, len(ts)-cut_window_len), high=len(ts)-1)
        sample_window = ts[max(0, cut_point - self.memory) : cut_point]
        sample[-len(sample_window):] = sample_window
        sample_mask[-len(sample_window):] = 1.0
        label = np.array(ts[cut_point])

        return sample, label, sample_mask


    def sample_ts(self, ts:np.array, freq:str) -> Tuple[np.array, np.array, np.array]:
        """Sampling a training sample a time series.

        Sampling algorithm:
            1. Sample sample_length in range min_length to length of model memory. Frequency dependent.
            2. Sample end index in range sample_length to length of time series.
            3. Create the sample by choosing the observations in range [(end_index - sample_length), end_index]
            4. Use the end_index entry of the time series as label.

        Args:
            ts (np.array): A time series in the form of an array.
            freq (str): The frequency of the time series. One capitalized letter.

        Returns:
            Tuple[np.array, np.array, np.arrray]: The sample, label and mask.
        """
        min_length = self.min_lengths[freq]
        #max_length = min(self.memory, len(ts))
        max_length = min(self.memory, len(ts)) + 1
        assert min_length < max_length, f"min_length ({min_length}) longer than max_length ({max_length})"

        length = np.random.randint(min_length, max_length) - 1
        assert length < len(ts), f"length ({length}) longer than len(ts) ({len(ts)})"
        end_index = np.random.randint(length, len(ts))

        sample = ts[(end_index - length) : end_index]
        label = np.array(ts[end_index])

        sample_mask = np.zeros(self.memory)
        sample_mask[-len(sample):] = 1.0

        sample = np.pad(sample, (self.memory - len(sample), 0), mode="constant", constant_values=0)

        assert len(sample) == self.memory, f"Sample is not as long as memory. Length sample: {len(sample)}, memory:{self.memory}"
        assert len(sample_mask) == self.memory, f"Mask is not as long as memory. Length mask: {len(sample_mask)}, memory:{self.memory}"
    
        return sample, label, sample_mask

    def prepare_ts(self, ts:Dict) -> Dict:
        """Prepare a preprocessed time series for training. 
        Use only values before a cutoff date. 
        Storing observations as floats in a numpy array. 
        Scaling the time series using the full time series. Required to stabilize training with some Scalers. The best would be to scale each sample by itself.

        Args:
            ts (Dict): A time series with observations and metadata. Needs frequency and observations.

        Returns:
            Dict: The same time series with values from before the cutoff date as floats in a numpy array and the frequency as a string.
        """
        if self.cutoff_date is not None:
            vals = []
            for obs in ts["observations"]:
                if time.strptime(obs["date"], "%Y-%m-%d") < self.cutoff_date_date:
                    vals.append(float(obs["value"])) 
            vals = np.array(vals)
        else:
            vals = np.array([float(obs["value"]) for obs in ts["observations"]])
    
        if self.ds_type == "train":
            # remove the last horizon of each time series if the dataset is train
            h = self.horizons[ts["frequency"]]
            if len(vals) < h:
                vals = np.array([])
            else:
                vals = vals[:-h]
        if np.count_nonzero(vals) < 0.5 * len(vals):
            vals = np.array([])
        elif len(vals[vals==np.mean(vals)]) == len(vals):
            vals = np.array([])
        elif len(vals[vals < 0]) > 0:
            vals = np.array([])
    
        if len(vals) < self.min_lengths[ts["frequency"]]:
            vals = np.array([])
        
        prepared_ts = {
            "frequency": ts["frequency"],
            "values" : vals
        }

        return prepared_ts
        


