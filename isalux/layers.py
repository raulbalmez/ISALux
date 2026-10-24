"""Building blocks of ISALux: HISA-MSA (Sec. 3.3), MoE-FFN (Sec. 3.3) and the
Transformer block (Eqs. 5-6)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channels of every pixel."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class LoRA(nn.Module):
    """Low-rank adaptation: delta = (X . alpha) . beta, alpha in R^{C x C/r},
    beta in R^{C/r x C}. beta starts at zero so training starts from the plain
    projection."""

    def __init__(self, dim, r=4):
        super().__init__()
        self.alpha = nn.Linear(dim, max(dim // r, 1), bias=False)
        self.beta = nn.Linear(max(dim // r, 1), dim, bias=False)
        nn.init.zeros_(self.beta.weight)

    def forward(self, tokens):  # (B, HW, C) -> (B, HW, C)
        return self.beta(self.alpha(tokens))


class SelfAttention(nn.Module):
    """Multi-headed self-attention with a 3x3 QKV projection, LoRA on Q/K/V
    (Eq. 7) and a learnable temperature (Eq. 8). Attention is computed across
    channels (C x C map, Fig. 2), as in Retinexformer."""

    def __init__(self, dim, heads, lora_rank=4):
        super().__init__()
        self.heads = heads
        self.proj = nn.Conv2d(dim, dim * 3, 3, padding=1, bias=False)
        self.lora_q = LoRA(dim, lora_rank)
        self.lora_k = LoRA(dim, lora_rank)
        self.lora_v = LoRA(dim, lora_rank)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.proj(x).chunk(3, dim=1)
        tokens = x.flatten(2).transpose(1, 2)                       # (B, HW, C)
        q = q.flatten(2) + self.lora_q(tokens).transpose(1, 2)      # (B, C, HW)
        k = k.flatten(2) + self.lora_k(tokens).transpose(1, 2)
        v = v.flatten(2) + self.lora_v(tokens).transpose(1, 2)
        split = lambda t: t.reshape(b, self.heads, c // self.heads, h * w)
        q, k, v = split(q), split(k), split(v)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / self.temperature         # (B,h,c',c')
        out = attn.softmax(dim=-1) @ v
        return out.reshape(b, c, h, w)


class HISAMSA(nn.Module):
    """Hybrid Illumination and Semantics-Aware MSA (Eq. 9):
        M_E = v * (M^i . F_Pi) + w * (M^s . F_Ps)
    with two parallel attention modules and learnable scalars v, w."""

    def __init__(self, dim, heads, lora_rank=4):
        super().__init__()
        self.attn_illum = SelfAttention(dim, heads, lora_rank)
        self.attn_sem = SelfAttention(dim, heads, lora_rank)
        self.v = nn.Parameter(torch.ones(1))
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x, f_illum, f_sem):
        return (self.v * self.attn_illum(x) * f_illum +
                self.w * self.attn_sem(x) * f_sem)


class Expert(nn.Module):
    """E_i(F) = conv1x1(GELU(conv3x3(GELU(conv1x1(F)))))  (Eq. 10)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim, 1))

    def forward(self, x):
        return self.net(x)


class MoEFFN(nn.Module):
    """Sparse mixture-of-experts feed-forward network (Eqs. 11-12).

    G = softmax(W_g . avgpool(F)); the top-k experts (k=2 of N=4) are run and
    their outputs are summed with weights p(E_i | F). Gating is per image.
    """

    def __init__(self, dim, hidden, num_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(Expert(dim, hidden)
                                     for _ in range(num_experts))

    def forward(self, x):
        probs = self.gate(x.mean(dim=(2, 3))).softmax(dim=-1)       # (B, N)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            chosen = top_i == e                                     # (B, k)
            rows = chosen.any(dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            weight = (top_p * chosen).sum(dim=1)[rows].view(-1, 1, 1, 1)
            out = out.index_add(0, rows, weight * expert(x[rows]))
        return out


class TransformerBlock(nn.Module):
    """Eqs. (5)-(6):
        F'   = F + LN(HISA-MSA(F, F_Pi, F_Ps))
        Fout = F' + LN(MoE-FFN(F'))
    """

    def __init__(self, dim, heads, ffn_mult=1.0, num_experts=4, top_k=2,
                 lora_rank=4):
        super().__init__()
        self.msa = HISAMSA(dim, heads, lora_rank)
        self.norm1 = LayerNorm2d(dim)
        self.ffn = MoEFFN(dim, int(dim * ffn_mult), num_experts, top_k)
        self.norm2 = LayerNorm2d(dim)

    def forward(self, x, f_illum, f_sem):
        x = x + self.norm1(self.msa(x, f_illum, f_sem))
        return x + self.norm2(self.ffn(x))
