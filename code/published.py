"""Published inference pieces for ThyroHybridNet.

Asymmetric gland-nodule guidance, inverse-root area weighting,
four-scale flip averaging, and the soft gland gate.
This file does not train a model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NODULE_SCALES = (0.75, 1.0, 1.25, 1.5)


class AGNG(nn.Module):
    """Gland amplifies nodule. Nodule can only boost gland."""

    def __init__(self, channels):
        super().__init__()
        self.gland_to_nodule = nn.Conv2d(channels, channels, kernel_size=1)
        self.nodule_to_gland = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, gland, nodule):
        nodule_out = nodule + nodule * torch.sigmoid(self.gland_to_nodule(gland))
        gland_out = gland + gland * (1.0 + torch.sigmoid(self.nodule_to_gland(nodule)))
        return gland_out, nodule_out


def iraw_weight(area_px, size=512, cap=8.0):
    """Per-image Dice weight. Smaller foregrounds receive a larger weight, capped at 8."""
    fraction = area_px / float(size * size)
    raw = 1.0 / torch.sqrt(fraction + 1e-4)
    return raw.clamp(1.0, cap)


def flip_average(predict_prob, image):
    """Mean of a probability map and its horizontal flip."""
    flipped = torch.flip(image, dims=(-1,))
    back = torch.flip(predict_prob(flipped), dims=(-1,))
    return 0.5 * (predict_prob(image) + back)


def tssa(predict_prob, image, scales=NODULE_SCALES):
    """Nodule inference: average flip-pairs over four scales, then resize to the input."""
    target = image.shape[-2:]
    total = None
    for scale in scales:
        if scale == 1.0:
            view = image
        else:
            view = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
        pair = flip_average(predict_prob, view)
        if pair.shape[-2:] != target:
            pair = F.interpolate(pair, size=target, mode="bilinear", align_corners=False)
        total = pair if total is None else total + pair
    return total / float(len(scales))


def soft_gland_gate(nodule_prob, gland_prob, kernel=15, tau_min=0.05):
    """Suppress nodule probability outside a dilated gland response."""
    pad = kernel // 2
    pooled = F.max_pool2d(gland_prob, kernel_size=kernel, stride=1, padding=pad)
    return nodule_prob * pooled.clamp(min=tau_min)
