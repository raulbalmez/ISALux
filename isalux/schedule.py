import math


def isalux_lr(it, total=300_000, lr0=2e-4, lr_peak=3e-4, lr_mid=2e-4,
              lr_min=1e-6, t_peak=92_000, t_mid=208_000):
    """Learning-rate schedule of Sec. 4.1: 2e-4 rising to 3e-4 at 92k
    iterations, cosine decay to 2e-4 at 208k, then cosine decay to 1e-6."""
    if it < t_peak:
        return lr0 + (lr_peak - lr0) * it / t_peak
    cos = lambda a, b, t: b + (a - b) * 0.5 * (1 + math.cos(math.pi * t))
    if it < t_mid:
        return cos(lr_peak, lr_mid, (it - t_peak) / (t_mid - t_peak))
    return cos(lr_mid, lr_min, min((it - t_mid) / (total - t_mid), 1.0))
