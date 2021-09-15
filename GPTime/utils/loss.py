import torch

def divide_non_nan(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """x/y where resulting NaN or Inf are replaced by 0. 
    https://github.com/ElementAI/N-BEATS/blob/04f56c4ca4c144071b94089f7195b1dd606072b0/common/torch/ops.py#L38

    Args:
        x (torch.Tensor): [description]
        y (torch.Tensor): [description]

    Returns:
        torch.Tensor: [description]
    """
    res = x / y
    res[torch.isnan(res)] = 0.0
    res[torch.isinf(res)] = 0.0
    return res

 
def smape_loss(forecast: torch.Tensor, target: torch.Tensor, tmp1:None, mask:torch.Tensor, tmp2:None) -> torch.Tensor:
    """Measures the Symmetric Mean Absolute Percentage Error. https://robjhyndman.com/hyndsight/smape/

    Args:
        forecast (torch.Tensor): The forecasted value(s)
        target (torch.Tensor): The target value(s)
        mask (torch.Tensor): The mask indicating potentially padded zeros in the forecast.

    Returns:
        torch.Tensor: The loss.
    """
    return 200 * torch.mean(
        divide_non_nan(
            torch.abs(forecast - target), torch.abs(forecast.data) + torch.abs(target.data)
            )
        )
    

def mase_loss(forecast, target, sample, sample_mask, frequency):
    """The Mean Absolute Scaled Error.
    TODO: Fix the naive seasonal scaling for batches containing different frequencies.
    Args:
        forecast (torch.Tensor): The forecasted value(s)
        target (torch.Tensor): The target value(s)
        sample (torch.Tensor): The insample values used to calculate the scaling.
        sample_mask (torch.Tensor): The mask indicating potentially padded zeros in the forecast.
        frequency (int): The frequency of the data used to scale by the naive seasonal forecast.

    Returns:
        torch.Tensor: The loss.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaling = []
    #np_in_sample =
    for i, row in enumerate(sample):
        scaling.append(torch.mean(torch.abs(row[frequency[i]:] - row[:-frequency[i]]), dim=0))
    scaling = torch.tensor(scaling).to(device)
    #scaling2 = torch.mean(torch.abs(sample[:, frequency:] - sample[:, :-frequency]), dim=1)
    #scaling2 = torch.mean(torch.abs(sample[:, 12:] - sample[:, :-12]), dim=1)
    #scaling = torch.mean(torch.abs(sample[:, 12:] - sample[:, :-12]), dim=1)
    #assert torch.sum(scaling - scaling2) == 0, "not 0 scaling and scaling2"
    inv_scaling_masked = divide_non_nan(sample_mask, scaling.unsqueeze(1))

    return torch.mean(torch.abs(target - forecast) * inv_scaling_masked)
