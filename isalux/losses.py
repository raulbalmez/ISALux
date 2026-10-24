"""Hybrid loss (Sec. 3.4): weighted sum of L2 (Eq. 13), VGG-19 perceptual
loss (Eq. 14, left) and MS-SSIM loss (Eq. 14, right)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchvision.models import VGG19_Weights, vgg19


class PerceptualLoss(nn.Module):
    """L1 distance between frozen VGG-19 features (up to conv5_4)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.features = vgg19(weights=weights).features[:35].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        norm = lambda x: (x - self.mean) / self.std
        return F.l1_loss(self.features(norm(pred)), self.features(norm(target)))


class HybridLoss(nn.Module):
    """L = w_l2 * L2 + w_perc * L_perc + w_ssim * L_MS-SSIM.
    Set a weight to 0 to ablate a term (Table 3)."""

    def __init__(self, w_l2=1.0, w_perc=0.05, w_ssim=0.5, pretrained=True):
        super().__init__()
        self.w_l2, self.w_perc, self.w_ssim = w_l2, w_perc, w_ssim
        self.perc = PerceptualLoss(pretrained) if w_perc > 0 else None

    def forward(self, pred, target):
        loss = pred.new_zeros(())
        if self.w_l2 > 0:
            loss = loss + self.w_l2 * F.mse_loss(pred, target)
        if self.w_perc > 0:
            loss = loss + self.w_perc * self.perc(pred, target)
        if self.w_ssim > 0:
            loss = loss + self.w_ssim * (
                1.0 - ms_ssim(pred.clamp(0, 1), target, data_range=1.0))
        return loss
