"""Illumination prior (Sec. 3.1) and semantic prior (Sec. 3.2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large)

NUM_CLASSES = 21


def illumination_prior(img):
    """P^i = 1 - max_c(I)  (Eq. 1), shape (B, 1, H, W)."""
    return 1.0 - img.max(dim=1, keepdim=True)[0]


def illumination_pyramid(p_i, levels=3):
    """P^i_0, P^i_1, P^i_2 = P^i, down(P^i, 2), down(P^i, 4)  (Eq. 2)."""
    return [p_i if k == 0 else F.avg_pool2d(p_i, 2 ** k) for k in range(levels)]


class SemanticPrior(nn.Module):
    """Frozen MobileNetV3 (DeepLabV3-MobileNetV3-Large, 11.03M parameters,
    trained on a COCO subset with the 21 VOC classes). Returns per-class
    probability maps P^s in R^{H x W x 21}  (Eq. 3)."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = (DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
                   if pretrained else None)
        self.net = deeplabv3_mobilenet_v3_large(weights=weights,
                                                weights_backbone=None,
                                                aux_loss=None if pretrained
                                                else False)
        self.net.aux_classifier = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode=True):  # always frozen / eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, img):
        logits = self.net((img - self.mean) / self.std)["out"]
        return logits.softmax(dim=1)
