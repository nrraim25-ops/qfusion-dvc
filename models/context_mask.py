"""Differentiable Context Mask Module for Temporal Event Soft-Pooling.

Per Section 7.8:
- For each event with predicted boundaries [start, end] in [0, 1]:
  Computes a smooth differentiable temporal sigmoid mask over snippets t in [0, T-1]:
  M_{i, t} = sigmoid(temperature * (p_t - s_i)) * sigmoid(temperature * (e_i - p_t))
- Performs soft-pooling across encoder representations to produce a 512-dim context vector per event.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableContextMask(nn.Module):
    def __init__(self, embed_dim: int = 512, temperature: float = 10.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        # Learnable temperature refinement
        self.temp_scale = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, encoder_out: torch.Tensor, pred_segments: torch.Tensor) -> torch.Tensor:
        """Computes soft-pooled context vectors for each predicted event.
        
        Args:
            encoder_out: Tensor of shape [B, T, 512]
            pred_segments: Tensor of shape [B, num_queries, 2] in [start, end] in [0, 1]
        Returns:
            context_vectors: Tensor of shape [B, num_queries, 512]
            masks: Tensor of shape [B, num_queries, T]
        """
        B, T, D = encoder_out.shape
        _, num_queries, _ = pred_segments.shape
        device = encoder_out.device

        # Normalized temporal coordinates of sequence snippets in [0, 1]
        p_t = torch.linspace(0.5 / T, 1.0 - 0.5 / T, T, device=device).view(1, 1, T) # [1, 1, T]

        s = pred_segments[..., 0].unsqueeze(-1) # [B, num_queries, 1]
        e = pred_segments[..., 1].unsqueeze(-1) # [B, num_queries, 1]

        # Smooth sigmoid boundaries
        tau = torch.clamp(self.temp_scale, min=1.0, max=50.0)
        mask_start = torch.sigmoid(tau * (p_t - s))
        mask_end = torch.sigmoid(tau * (e - p_t))
        raw_mask = mask_start * mask_end # [B, num_queries, T]

        # Normalize weights along temporal dimension T
        weights = raw_mask / (raw_mask.sum(dim=-1, keepdim=True) + 1e-6) # [B, num_queries, T]

        # Soft-pool context vectors: [B, num_queries, T] @ [B, T, D] -> [B, num_queries, D]
        context_vectors = torch.bmm(weights, encoder_out) # [B, num_queries, 512]

        return context_vectors, weights
