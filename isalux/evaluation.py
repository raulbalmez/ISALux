import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def pad_to_multiple(x, m=4):
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), h, w


def gt_mean_scale(pred, gt):
    """Retinexformer's GT-mean correction: rescale the prediction so that its
    mean grey level equals that of the ground truth, then clip to [0, 1]."""
    w = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    mp, mg = (pred * w).sum(1).mean(), (gt * w).sum(1).mean()
    return (pred * (mg / mp)).clamp(0, 1)


def psnr(pred, gt):
    return (-10 * torch.log10(F.mse_loss(pred, gt))).item()


@torch.no_grad()
def evaluate_model(model, loader, device, gt_mean=True):
    """Mean PSNR / SSIM over full-resolution test images."""
    model.eval()
    ps, ss = [], []
    for low, high in loader:
        low, high = low.to(device), high.to(device)
        x, h, w = pad_to_multiple(low)
        pred = model(x)[..., :h, :w].clamp(0, 1)
        if gt_mean:
            pred = gt_mean_scale(pred, high)
        ps.append(psnr(pred, high))
        ss.append(ssim(pred, high, data_range=1.0).item())
    return float(np.mean(ps)), float(np.mean(ss))
