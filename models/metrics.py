import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from sklearn.metrics import confusion_matrix, r2_score


def scalar_metrics(pred, true, eps=1e-8):
    pred = pred.detach().cpu().view(-1).numpy()
    true = true.detach().cpu().view(-1).numpy()
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / (np.abs(true) + eps))),
    }


def slip_area_metrics(logits, target):
    pred = torch.argmax(F.softmax(logits, dim=1), dim=1).detach().cpu().numpy()
    true = target.detach().cpu().numpy()
    accs, precisions, recalls, f1s, ious = [], [], [], [], []
    for p, t in zip(pred, true):
        tn, fp, fn, tp = confusion_matrix(t.ravel(), p.ravel(), labels=[0, 1]).ravel()
        accs.append((tp + tn) / (tp + tn + fp + fn + 1e-8))
        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (tp + fn + 1e-8))
        f1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1] + 1e-8))
        ious.append(tp / (tp + fp + fn + 1e-8))
    return {
        "area_acc": float(np.mean(accs)),
        "area_pre": float(np.mean(precisions)),
        "area_rec": float(np.mean(recalls)),
        "area_f1": float(np.mean(f1s)),
        "area_iou": float(np.mean(ious)),
    }


def _safe_nanmean(values):
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    if np.isnan(array).all():
        return float("nan")
    return float(np.nanmean(array))


def _spatial_corr_coeff(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.size == 0 or true.size == 0 or np.var(pred) == 0 or np.var(true) == 0:
        return float("nan")
    return float(np.corrcoef(pred.reshape(-1), true.reshape(-1))[0, 1])


def _psnr(mse, data_range=1.0):
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(data_range / np.sqrt(mse)))


def slip_distribution_metrics(pred, true, mask=None):
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() if mask is not None else np.ones_like(true_np, dtype=np.float32)
    mses, maes, r2s, ssims, sccs, psnrs = [], [], [], [], [], []
    for p, t, m in zip(pred_np, true_np, mask_np):
        p_item = np.squeeze(p)
        t_item = np.squeeze(t)
        m_item = np.squeeze(m)
        active = m_item > 0
        if not np.any(active):
            continue
        p_active = p_item[active]
        t_active = t_item[active]
        mses.append(np.mean((p_active - t_active) ** 2))
        maes.append(np.mean(np.abs(p_active - t_active)))
        r2s.append(r2_score(t_active, p_active) if np.var(t_active) > 0 else np.nan)

        p_masked = p_item * active.astype(np.float32)
        t_masked = t_item * active.astype(np.float32)
        data_range = float(np.max(t_masked) - np.min(t_masked))
        if not np.isfinite(data_range) or data_range <= 0:
            data_range = 1.0

        p_image = torch.tensor(p_masked, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_image = torch.tensor(t_masked, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ssims.append(ssim(p_image, t_image, data_range=data_range, size_average=True).item())
        sccs.append(_spatial_corr_coeff(p_masked, t_masked))
        psnrs.append(_psnr(np.mean((p_masked - t_masked) ** 2), data_range=data_range))
    return {
        "slip_mse": _safe_nanmean(mses),
        "slip_mae": _safe_nanmean(maes),
        "slip_r2": _safe_nanmean(r2s),
        "slip_ssim": _safe_nanmean(ssims),
        "slip_scc": _safe_nanmean(sccs),
        "slip_psnr": _safe_nanmean(psnrs),
    }
