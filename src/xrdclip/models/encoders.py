"""Dual encoders for XRD-CLIP.

  DiffractionEncoder : 1D ResNet over a diffractogram (5-50 deg, 0.02 deg -> 2251 pts) -> R^d
  StructureEncoder   : atom-set Transformer over (element, fractional coords) + lattice -> R^d

Both project to a shared L2-normalised embedding space where the contrastive loss lives.

Design notes that matter:
  * The diffraction side is a CNN, not a transformer, because a diffractogram is a translation-
    sensitive 1D signal where local peak shape carries the physics; a wide-kernel CNN is the right
    inductive bias and trains in minutes (SPEC §2.1).
  * The structure side gets the LATTICE explicitly. Peak positions are a pure function of the cell,
    so a structure encoder that never sees the cell cannot place peaks -- it is information-starved
    by construction. The atom set fixes intensities; the lattice fixes positions (SPEC §2.2).
  * We use an atom-set Transformer rather than a full periodic message-passing GNN: it is faithful
    to the 3D structure (elements + positions + cell), far lighter to train in a 3-week sprint, and
    the periodic-GNN variant is a clean ablation rather than a prerequisite.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_Z = 100  # atomic-number embedding table size (we filter to Z<=53 = iodine, but leave headroom)


# ---------------------------------------------------------------------------------------------
# diffraction encoder
# ---------------------------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, k: int = 7):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv1d(cin, cout, k, stride=stride, padding=p, bias=False)
        self.gn1 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv1d(cout, cout, k, padding=p, bias=False)
        self.gn2 = nn.GroupNorm(8, cout)
        self.short = (
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.GroupNorm(8, cout))
            if stride != 1 or cin != cout else nn.Identity()
        )

    def forward(self, x):
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(h + self.short(x))


class DiffractionEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width, 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
        )
        chans = [width, width, width * 2, width * 2, width * 4, width * 4]
        blocks = []
        for i in range(len(chans) - 1):
            blocks.append(ResBlock1D(chans[i], chans[i + 1], stride=2 if i % 2 == 0 else 1))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(chans[-1], embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):                      # x: (B, 1, N_POINTS)
        h = self.blocks(self.stem(x))
        h = self.pool(h).squeeze(-1)
        return F.normalize(self.proj(h), dim=-1)


# ---------------------------------------------------------------------------------------------
# structure encoder
# ---------------------------------------------------------------------------------------------

class StructureEncoder(nn.Module):
    """Permutation-invariant Transformer over atoms, conditioned on the lattice.

    Input per crystal:
        atom_z    (B, A) int   atomic numbers, 0 = padding
        atom_frac (B, A, 3)    fractional coordinates in [0,1)
        lattice   (B, L)       cell descriptor (see dataset.structure_features)
        mask      (B, A) bool  True where a real atom sits
    """

    def __init__(self, embed_dim: int = 256, lattice_dim: int = 8, width: int = 256,
                 depth: int = 4, heads: int = 8):
        super().__init__()
        self.z_embed = nn.Embedding(MAX_Z, width, padding_idx=0)
        # Fractional coords are periodic; feed sin/cos so 0.0 and 1.0 map to the same place.
        self.pos_proj = nn.Linear(6, width)
        self.lattice_proj = nn.Sequential(nn.Linear(lattice_dim, width), nn.GELU(), nn.Linear(width, width))

        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 4, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.proj = nn.Sequential(
            nn.Linear(width * 2, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, atom_z, atom_frac, lattice, mask):
        two_pi = 2.0 * torch.pi
        pos = torch.cat([torch.sin(two_pi * atom_frac), torch.cos(two_pi * atom_frac)], dim=-1)
        h = self.z_embed(atom_z) + self.pos_proj(pos)                 # (B, A, W)

        # A lattice token, prepended so every atom can attend to the cell.
        lat = self.lattice_proj(lattice).unsqueeze(1)                # (B, 1, W)
        h = torch.cat([lat, h], dim=1)
        m = torch.cat([torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask], dim=1)

        h = self.transformer(h, src_key_padding_mask=~m)

        # Pool atoms (masked mean) and read the lattice token; concatenate both.
        lat_out = h[:, 0]
        atoms = h[:, 1:]
        am = mask.unsqueeze(-1).float()
        atom_mean = (atoms * am).sum(1) / am.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(torch.cat([lat_out, atom_mean], dim=-1)), dim=-1)
