"""The contrastive model: two encoders + a symmetric InfoNCE (CLIP) objective.

Positives are (augmented pattern of structure i, structure i). Negatives are the other structures
in the batch, so the batch size IS the negative count and matters a lot -- target B >= 512 (SPEC
§2.3). The temperature is learnable but clamped so it cannot collapse.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xrdclip.models.encoders import DiffractionEncoder, StructureEncoder


class XRDClip(nn.Module):
    def __init__(self, embed_dim: int = 256, lattice_dim: int = 8):
        super().__init__()
        self.pattern_encoder = DiffractionEncoder(embed_dim)
        self.structure_encoder = StructureEncoder(embed_dim, lattice_dim)
        # log temperature, initialised at ln(1/0.07) as in CLIP; clamped in forward.
        self.log_temp = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def encode_pattern(self, pattern):
        return self.pattern_encoder(pattern)

    def encode_structure(self, atom_z, atom_frac, lattice, mask):
        return self.structure_encoder(atom_z, atom_frac, lattice, mask)

    def forward(self, pattern, atom_z, atom_frac, lattice, mask):
        zp = self.encode_pattern(pattern)
        zs = self.encode_structure(atom_z, atom_frac, lattice, mask)
        temp = self.log_temp.clamp(max=np.log(1 / 0.01)).exp()   # temp <= 100, i.e. tau >= 0.01
        logits = temp * zp @ zs.T                                # (B, B)
        return logits, zp, zs


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy; the correct match for row/column i is index i."""
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


@torch.no_grad()
def batch_topk_accuracy(logits: torch.Tensor, ks=(1, 5)) -> dict:
    """In-batch retrieval accuracy — a fast training-time proxy for the real eval."""
    target = torch.arange(logits.shape[0], device=logits.device)
    order = logits.argsort(dim=1, descending=True)
    ranks = (order == target[:, None]).float().argmax(dim=1)
    return {f"top{k}": float((ranks < k).float().mean()) for k in ks}
