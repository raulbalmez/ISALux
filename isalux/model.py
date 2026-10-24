"""ISALux / ISA-T (Fig. 1): a 2-level U-shaped Transformer with illumination
and semantic priors injected into every attention block."""
import torch
import torch.nn as nn

from .layers import TransformerBlock
from .priors import (NUM_CLASSES, SemanticPrior, illumination_pyramid,
                     illumination_prior)


class PriorAdapter(nn.Module):
    """Maps the priors of one level to the channel depth of its Transformer
    block (Sec. 3.3): a 1x1 conv for the illumination map (already at the
    right resolution) and a strided 3x3 conv (s in {1, 2, 4}) for the
    semantic map."""

    def __init__(self, dim, stride):
        super().__init__()
        self.illum = nn.Conv2d(1, dim, 1)
        self.sem = nn.Conv2d(NUM_CLASSES, dim, 3, stride=stride, padding=1)

    def forward(self, p_i, p_s):
        return self.illum(p_i), self.sem(p_s)


class ISAT(nn.Module):
    """ISA-T: I_E = ISA-T(I, P^s, P^i)  (Eq. 4)."""

    def __init__(self, dim=30, heads=(1, 2, 4), ffn_mult=0.4, num_experts=4,
                 top_k=2, lora_rank=4):
        super().__init__()
        d0, d1, d2 = dim, dim * 2, dim * 4
        blk = lambda d, h: TransformerBlock(d, h, ffn_mult, num_experts,
                                            top_k, lora_rank)
        self.embed = nn.Conv2d(3, d0, 3, padding=1)

        self.enc0, self.pri_e0 = blk(d0, heads[0]), PriorAdapter(d0, 1)
        self.down0 = nn.Conv2d(d0, d1, 3, stride=2, padding=1)
        self.enc1, self.pri_e1 = blk(d1, heads[1]), PriorAdapter(d1, 2)
        self.down1 = nn.Conv2d(d1, d2, 3, stride=2, padding=1)

        self.bot, self.pri_b = blk(d2, heads[2]), PriorAdapter(d2, 4)

        up = lambda i, o: nn.ConvTranspose2d(i, o, 3, stride=2, padding=1,
                                             output_padding=1)
        self.up1, self.fuse1 = up(d2, d1), nn.Conv2d(d1 * 2, d1, 3, padding=1)
        self.dec1, self.pri_d1 = blk(d1, heads[1]), PriorAdapter(d1, 2)
        self.up0, self.fuse0 = up(d1, d0), nn.Conv2d(d0 * 2, d0, 3, padding=1)
        self.dec0, self.pri_d0 = blk(d0, heads[0]), PriorAdapter(d0, 1)

        self.out = nn.Conv2d(d0, 3, 3, padding=1)

    def forward(self, img, p_s, p_i):
        pyr = illumination_pyramid(p_i, 3)
        x = self.embed(img)

        e0 = self.enc0(x, *self.pri_e0(pyr[0], p_s))
        e1 = self.enc1(self.down0(e0), *self.pri_e1(pyr[1], p_s))
        b = self.bot(self.down1(e1), *self.pri_b(pyr[2], p_s))

        d1 = self.fuse1(torch.cat([self.up1(b), e1], dim=1))
        d1 = self.dec1(d1, *self.pri_d1(pyr[1], p_s))
        d0 = self.fuse0(torch.cat([self.up0(d1), e0], dim=1))
        d0 = self.dec0(d0, *self.pri_d0(pyr[0], p_s))
        return self.out(d0)


class ISALux(nn.Module):
    """Full model: computes both priors from the low-light input and runs
    ISA-T. Input and output are RGB images in [0, 1]; H and W must be
    multiples of 4."""

    def __init__(self, pretrained_seg=True, **kwargs):
        super().__init__()
        self.semantic = SemanticPrior(pretrained_seg)
        self.isat = ISAT(**kwargs)

    def forward(self, img):
        p_i = illumination_prior(img)
        p_s = self.semantic(img)
        return self.isat(img, p_s, p_i)

    def train(self, mode=True):
        super().train(mode)
        self.semantic.eval()
        return self


def count_params(module):
    return sum(p.numel() for p in module.parameters())
