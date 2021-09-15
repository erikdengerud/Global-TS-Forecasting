import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import logging

logger = logging.getLogger(__name__)



class MLP(torch.nn.Module): 
    """
    MLP 
    """
    def __init__(
        self,
        in_features,
        out_features,
        num_layers,
        n_hidden,
        residual:str=None,
        res_block_size:int=1,
        forecast_horizon:int=1,
        skip_connections:bool=False,
        seasonal_naive:bool=False,
        bias:bool=True,
        dropout:float=0.2,
        encode_frequencies:bool=False,
        )->None:
        super(MLP, self).__init__()
        self.input_size = in_features
        self.output_size = out_features
        self.n_layers = num_layers
        self.layer_size = n_hidden
        self.res_block_size = res_block_size
        self.residual = residual
        self.frequency = 12
        self.skip_connections = skip_connections
        self.seasonal_naive = seasonal_naive
        self.memory = in_features
        self.dropout = dropout
        self.encode_frequencies = encode_frequencies

        self.one_hot = {
            "Y": np.array([1,0,0,0,0,0]),
            "Q": np.array([0,1,0,0,0,0]),
            "M": np.array([0,0,1,0,0,0]),
            "W": np.array([0,0,0,1,0,0]),
            "D": np.array([0,0,0,0,1,0]),
            "H": np.array([0,0,0,0,0,1]),
            "O": np.array([0,0,0,0,0,0]),
            "yearly": np.array([1,0,0,0,0,0]),
            "quarterly": np.array([0,1,0,0,0,0]),
            "monthly": np.array([0,0,1,0,0,0]),
            "weekly": np.array([0,0,0,1,0,0]),
            "daily": np.array([0,0,0,0,1,0]),
            "hourly": np.array([0,0,0,0,0,1]),
            }
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.layers = torch.nn.ModuleList()
        self.dropout_layers = torch.nn.ModuleList()
        if self.encode_frequencies:
            self.layers.append(torch.nn.Linear(in_features=in_features+6, out_features=n_hidden))
        else:
            self.layers.append(torch.nn.Linear(in_features=in_features, out_features=n_hidden))
        self.dropout_layers.append(nn.Dropout(p=dropout))
        for i in range(1, self.n_layers-1):
            self.layers.append(torch.nn.Linear(in_features=n_hidden, out_features=n_hidden))
            self.dropout_layers.append(nn.Dropout(p=dropout))
        self.out_layer = torch.nn.Linear(in_features=n_hidden, out_features=out_features)
 
        self.init_weights()
        #logger.debug(f"Building model with frequency {self.frequency}")
        logger.debug(f"Input size of model: {in_features}.")
        logger.debug(f"Number of layers: {num_layers}.")
        logger.debug(f"Number of hidden units: {n_hidden}.")
        logger.debug(f"Using frequency encoding: {encode_frequencies}")

    def forward(self, x, mask, last_period, freq_str_arr):
        naive = torch.gather(x, 1, last_period.unsqueeze(1))
        #naive = 0
        if self.skip_connections:
            skip = 0
        if self.encode_frequencies:
            try:
                one_hot_freq = []
                for f in freq_str_arr:
                    #logger.debug(f"f[0]: {f[0]}")
                    one_hot_freq.append(self.one_hot[f[0]])
                #logger.debug(f"len(one_hot_freq): {len(one_hot_freq)}")
                #logger.debug(f"one_hot_freq[:10]: {one_hot_freq[:3]}")
                ohf_arr = torch.from_numpy(np.array(one_hot_freq)).to(self.device).double()
                x = torch.cat((x, ohf_arr), 1)
            except Exception as e:
                logger.debug(e)
                logger.debug(f"len(one_hot_freq): {len(one_hot_freq)}")
                logger.debug(f"one_hot_freq[:10]: {one_hot_freq[:3]}")
                logger.debug(f"x.shape: {x.shape}")
                ohf_arr = np.array(one_hot_freq)
                logger.debug(f"ohf_arr.shape: {ohf_arr.shape}")
                ohf_tens = torch.from_numpy(ohf_arr)
                logger.debug(f"ohf_tens.shape: {ohf_tens.shape}")
                #for ohf in one_hot_freq:
                #    logger.debug(ohf)
                #logger.debug(f"ohf_arr.shape: {ohf_arr.shape}")
        x = self.layers[0](x)
        x = self.dropout_layers[0](x)
        res = x
        for i, layer in enumerate(self.layers[1:], start=1):
            if (i+1) % self.res_block_size:
                x = res + F.relu(layer(x)) # This is supposed to be better since the signal can pass directly through ref https://arxiv.org/abs/1603.05027
                x = self.dropout_layers[i](x)
                res = x
                if self.skip_connections:
                    skip += x
            else:
                x = F.relu(layer(x))
                x = self.dropout_layers[i](x)
        if self.skip_connections:
            skip = skip + x
            out = self.out_layer(skip)
        else:
            out = self.out_layer(x)
        forecast = out + naive

        return forecast

    def init_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.out_layer.weight)
