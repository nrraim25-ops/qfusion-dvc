"""Baseline Classical Fusion Module.

Per Section 7.3:
- Linear 768 -> 512 per modality
- Concatenate (1024)
- Linear 1024 -> 512
- LayerNorm(512)
Total parameters: approx 1.3M parameters.
"""

import torch
import torch.nn as nn


class ClassicalFusion(nn.Module):
    def __init__(self, video_dim: int = 768, audio_dim: int = 768, fused_dim: int = 512):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.fusion_linear = nn.Linear(fused_dim * 2, fused_dim)
        self.layernorm = nn.LayerNorm(fused_dim)
        self.relu = nn.ReLU()

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass for classical fusion.
        
        Args:
            video_feats: Tensor of shape [..., T, 768]
            audio_feats: Tensor of shape [..., T, 768]
        Returns:
            fused_feats: Tensor of shape [..., T, 512]
        """
        v_proj = self.relu(self.video_proj(video_feats)) # [..., T, 512]
        a_proj = self.relu(self.audio_proj(audio_feats)) # [..., T, 512]
        concat = torch.cat([v_proj, a_proj], dim=-1)     # [..., T, 1024]
        fused = self.fusion_linear(concat)               # [..., T, 512]
        out = self.layernorm(fused)                      # [..., T, 512]
        return out
