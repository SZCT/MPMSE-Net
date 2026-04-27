import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def ssim_loss(pred, target, data_range=1.0):
    return 1.0 - ssim(pred, target, data_range=data_range, size_average=True)


def weighted_average_slip_loss(pred, target, model_id, mu_area_cache):
    values_pred = []
    values_true = []
    for i in range(pred.size(0)):
        item = mu_area_cache[int(model_id[i].item())]
        mu = item["mu"].to(pred.device)
        area = item["area"].to(pred.device)
        weight_sum = item["mu_area_sum"].to(pred.device)
        values_pred.append((pred[i : i + 1] * mu * area).sum() / weight_sum)
        values_true.append((target[i : i + 1] * mu * area).sum() / weight_sum)
    return F.mse_loss(torch.stack(values_pred).view(-1), torch.stack(values_true).view(-1))
