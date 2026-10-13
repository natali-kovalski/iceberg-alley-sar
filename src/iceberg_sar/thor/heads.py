"""Heads on frozen THOR token features: a linear baseline and a CenterNet-style heatmap head.

Both map (B, C, h, w) features to (B, 1, h, w) logits, one berg score per token. Features are
standardized per channel with train-set stats first (an affine map, so the linear head stays
linear). Loss is CenterNet's penalty-reduced focal loss on the token-grid Gaussians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from iceberg_sar.config import Config

HEADS = ("linear", "centernet")


@dataclass(frozen=True)
class HeadParams:
    hidden: int
    dropout: float
    prior: float

    @classmethod
    def from_config(cls, cfg: Config) -> HeadParams:
        h = cfg.section("thor")["head"]
        return cls(hidden=int(h["hidden"]), dropout=float(h["dropout"]), prior=float(h["prior"]))


class Standardize(nn.Module):
    """Fixed per-channel (x - mean) / std; stats travel with the checkpoint."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("std", torch.ones(1, channels, 1, 1))

    def set_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view_as(self.mean))
        self.std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32)
                       .view_as(self.std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std


class LinearHead(nn.Module):
    def __init__(self, channels: int, p: HeadParams) -> None:
        super().__init__()
        self.norm = Standardize(channels)
        self.out = nn.Conv2d(channels, 1, 1)
        nn.init.constant_(self.out.bias, -math.log((1 - p.prior) / p.prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.norm(x))


class CenterNetHead(nn.Module):
    """Per-token MLP, one 3x3 conv for neighbourhood context, then one logit per token."""

    def __init__(self, channels: int, p: HeadParams) -> None:
        super().__init__()
        self.norm = Standardize(channels)
        self.body = nn.Sequential(
            nn.Dropout2d(p.dropout),
            nn.Conv2d(channels, p.hidden, 1), nn.GELU(),
            nn.Conv2d(p.hidden, p.hidden, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Conv2d(p.hidden, 1, 1)
        nn.init.constant_(self.out.bias, -math.log((1 - p.prior) / p.prior))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(self.norm(x)))


class Ensemble(nn.Module):
    """Mean probability of several heads, returned as a logit so callers stay the same."""

    def __init__(self, members: list[nn.Module]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.stack([torch.sigmoid(m(x)) for m in self.members]).mean(dim=0)
        return torch.logit(p.clamp(1e-6, 1 - 1e-6))


def build_head(kind: str, channels: int, p: HeadParams) -> nn.Module:
    if kind == "linear":
        return LinearHead(channels, p)
    if kind == "centernet":
        return CenterNetHead(channels, p)
    raise ValueError(f"Unknown head {kind!r}; choose from {HEADS}")


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
    alpha: float = 2.0, beta: float = 4.0,
) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss, summed over weighted tokens / number of peaks.

    `target` is the Gaussian heatmap (peaks exactly 1); `weight` 0 drops a token.
    Negatives near a peak are down-weighted by (1 - target)^beta.
    """
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1.0).float() * weight
    neg = (1.0 - (target >= 1.0).float()) * weight
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    return -(pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def extract_peaks(
    prob: np.ndarray, score_min: float, valid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima (3x3) at or above `score_min` -> ((n, 2) token (i, j), scores), best first.

    Plateaus (equal neighbours) keep one token, the first in raster order.
    """
    p = np.where(np.isfinite(prob), prob, -np.inf)
    if valid is not None:
        p = np.where(valid, p, -np.inf)
    is_max = (p == ndimage.maximum_filter(p, size=3, mode="constant", cval=-np.inf)) & (
        p >= score_min)
    ij = np.argwhere(is_max)
    # Drop later tokens of a plateau: a peak whose earlier 3x3 neighbour is also a peak.
    keep = np.ones(len(ij), bool)
    taken = set()
    for n, (i, j) in enumerate(ij):
        if any((i + di, j + dj) in taken for di in (-1, 0, 1) for dj in (-1, 0, 1)):
            keep[n] = False
        else:
            taken.add((i, j))
    ij = ij[keep]
    scores = p[ij[:, 0], ij[:, 1]] if len(ij) else np.empty(0)
    order = np.argsort(-scores, kind="stable")
    return ij[order], scores[order]
