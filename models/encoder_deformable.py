"""Deformable + Sparse-Attention Temporal Encoder.

Per Section 7.6:
- 1D Temporal Deformable Attention with K=4 sampling points per query
- Multi-head attention (e.g. M=4 heads, head_dim=128, embed_dim=512)
- Multi-layer transformer encoder (4 layers)
- Sparse pruning to 50% token keep ratio after layer 2 based on learned temporal saliency
- Residual connections, LayerNorm, and MLP blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableTemporalAttention1D(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, num_points: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Sampling offsets: [num_heads * num_points]
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points)
        # Attention weights: [num_heads * num_points]
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets near zero
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # Initial grid offsets along temporal dimension
        grid_init = torch.linspace(-0.1, 0.1, self.num_points).repeat(self.num_heads)
        self.sampling_offsets.bias.data = grid_init

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Forward pass for 1D deformable attention.
        
        Args:
            query: Tensor of shape [B, T_q, embed_dim]
            key_value: Tensor of shape [B, T_k, embed_dim]
        Returns:
            output: Tensor of shape [B, T_q, embed_dim]
        """
        B, T_q, _ = query.shape
        _, T_k, _ = key_value.shape

        # 1. Project values: [B, T_k, num_heads, head_dim]
        v = self.value_proj(key_value).view(B, T_k, self.num_heads, self.head_dim)

        # 2. Reference points along temporal sequence: [T_q] in range [0, 1]
        ref_points = torch.linspace(0.5 / T_q, 1.0 - 0.5 / T_q, T_q, device=query.device, dtype=query.dtype)
        ref_points = ref_points.view(1, T_q, 1, 1).repeat(B, 1, self.num_heads, self.num_points)

        # 3. Offsets: [B, T_q, num_heads, num_points]
        offsets = torch.tanh(self.sampling_offsets(query)).view(B, T_q, self.num_heads, self.num_points)
        # Normalized sampling locations in range [-1, 1] for grid_sample
        sampling_locations = (ref_points + offsets / T_k) * 2.0 - 1.0

        # 4. Attention weights: softmax over sampling points
        attn_weights = self.attention_weights(query).view(B, T_q, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 5. 1D Sampling using grid_sample
        # Reshape v to [B * num_heads, head_dim, 1, T_k]
        v_grid = v.permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, 1, T_k)
        
        # Reshape sampling_locations to [B * num_heads, T_q, num_points, 2]
        # x-coordinate is temporal location, y-coordinate is 0
        loc_x = sampling_locations.permute(0, 2, 1, 3).reshape(B * self.num_heads, T_q, self.num_points)
        loc_y = torch.zeros_like(loc_x)
        grid = torch.stack([loc_x, loc_y], dim=-1) # [B*H, T_q, K, 2]

        sampled = F.grid_sample(v_grid, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # sampled shape: [B * num_heads, head_dim, T_q, num_points]
        sampled = sampled.permute(0, 2, 3, 1) # [B * num_heads, T_q, num_points, head_dim]
        sampled = sampled.view(B, self.num_heads, T_q, self.num_points, self.head_dim)
        sampled = sampled.permute(0, 2, 1, 3, 4) # [B, T_q, num_heads, num_points, head_dim]

        # 6. Weighted aggregation across K sampling points
        weighted = (sampled * attn_weights.unsqueeze(-1)).sum(dim=3) # [B, T_q, num_heads, head_dim]
        weighted = weighted.reshape(B, T_q, self.embed_dim)

        return self.output_proj(weighted)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, num_points: int = 4, ffn_dim: int = 1024):
        super().__init__()
        self.self_attn = DeformableTemporalAttention1D(embed_dim, num_heads, num_points)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual block
        attn_out = self.self_attn(self.norm1(x), self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class DeformableSparseEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        num_points: int = 4,
        prune_ratio: float = 0.5
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.prune_ratio = prune_ratio

        # First 2 layers
        self.layer1 = DeformableEncoderLayer(embed_dim, num_heads, num_points)
        self.layer2 = DeformableEncoderLayer(embed_dim, num_heads, num_points)

        # Saliency / importance scoring for 50% sparsification after Layer 2
        self.saliency_scorer = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Subsequent layers (operating on pruned sequence)
        self.layer3 = DeformableEncoderLayer(embed_dim, num_heads, num_points)
        self.layer4 = DeformableEncoderLayer(embed_dim, num_heads, num_points)

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        """Forward pass through deformable + sparse attention encoder.
        
        Args:
            x: Tensor of shape [B, T, embed_dim] or [T, embed_dim]
        Returns:
            encoder_out: Tensor of shape [B, T, embed_dim] or [T, embed_dim]
            saliency_scores: Tensor of shape [B, T]
            keep_indices: Tensor of indices retained after pruning
        """
        is_unbatched = (x.ndim == 2)
        if is_unbatched:
            x = x.unsqueeze(0)

        B, T, D = x.shape

        # 1. Layers 1 and 2
        h = self.layer1(x)
        h = self.layer2(h)

        # 2. Saliency scoring & 50% pruning after layer 2
        scores = self.saliency_scorer(h).squeeze(-1) # [B, T]
        k_keep = max(1, int(math.ceil(T * (1.0 - self.prune_ratio))))

        # Select top-k salient snippet positions
        _, topk_idx = torch.topk(scores, k=k_keep, dim=-1, sorted=True) # [B, k_keep]
        # Sort indices temporally to preserve chronological sequence order
        topk_idx, _ = torch.sort(topk_idx, dim=-1)

        # Gather pruned tokens
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        h_pruned = torch.gather(h, dim=1, index=idx_expanded) # [B, k_keep, D]

        # 3. Layers 3 and 4 on pruned representations
        h_out = self.layer3(h_pruned)
        h_out = self.layer4(h_out)
        h_out = self.final_norm(h_out)

        # 4. Scatter back to full temporal length T with residual blending
        full_out = h.clone()
        full_out.scatter_(dim=1, index=idx_expanded, src=h_out)

        if is_unbatched:
            full_out = full_out.squeeze(0)
            scores = scores.squeeze(0)
            topk_idx = topk_idx.squeeze(0)

        return full_out, scores, topk_idx
