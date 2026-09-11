"""Parameter-Matched Classical Control Fusion Module.

Per Section 7.4:
- Shared compression: Linear(768, 3) per modality -> concat into 6-vector
- Processing block (matched parameter budget):
  4 layers of Linear(6, 3, bias=False) each.
  Each layer: 6 * 3 = 18 parameters.
  Total processing trainable parameters: 4 * 18 = 72 parameters EXACTLY.
  Structural coupling: delta = tanh(Linear_6to3(state))
  state[0:3] += delta, state[3:6] += delta (fixed untrainable cross-modal coupling analogous to CNOTs)
- Shared expansion: Linear(6, 512) + LayerNorm(512).
"""

import torch
import torch.nn as nn


class MatchedClassicalProcessingBlock(nn.Module):
    def __init__(self, num_layers: int = 4, in_dim: int = 6, hidden_dim: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(num_layers)
        ])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Processes the 6-dim state across 4 layers with tanh non-linearity and fixed cross-modal coupling.
        
        Args:
            state: Tensor of shape [..., 6]
        Returns:
            processed_state: Tensor of shape [..., 6]
        """
        for layer in self.layers:
            # delta has shape [..., 3]
            delta = torch.tanh(layer(state))
            # Fixed untrainable coupling between video (0:3) and audio (3:6) halves
            # (structural analog to quantum CNOT cross-modal entanglement)
            first_half = state[..., 0:3] + delta
            second_half = state[..., 3:6] + delta
            state = torch.cat([first_half, second_half], dim=-1)
        return state


class ClassicalMatchedFusion(nn.Module):
    def __init__(self, video_dim: int = 768, audio_dim: int = 768, fused_dim: int = 512):
        super().__init__()
        # Shared compression infrastructure (identical to QuantumFusion)
        self.video_compress = nn.Linear(video_dim, 3)
        self.audio_compress = nn.Linear(audio_dim, 3)

        # 72-parameter matched processing module
        self.processing = MatchedClassicalProcessingBlock(num_layers=4, in_dim=6, hidden_dim=3)

        # Shared expansion infrastructure (identical to QuantumFusion)
        self.expand = nn.Linear(6, fused_dim)
        self.layernorm = nn.LayerNorm(fused_dim)

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass for parameter-matched classical control fusion.
        
        Args:
            video_feats: Tensor of shape [..., T, 768]
            audio_feats: Tensor of shape [..., T, 768]
        Returns:
            fused_feats: Tensor of shape [..., T, 512]
        """
        # 1. Compress
        v_comp = self.video_compress(video_feats) # [..., T, 3]
        a_comp = self.audio_compress(audio_feats) # [..., T, 3]

        # 2. Concatenate into 6-vector
        state = torch.cat([v_comp, a_comp], dim=-1) # [..., T, 6]

        # 3. Process via 72-parameter matched classical circuit analog
        processed_state = self.processing(state) # [..., T, 6]

        # 4. Expand
        fused = self.expand(processed_state) # [..., T, 512]
        out = self.layernorm(fused)          # [..., T, 512]
        return out
