"""Event-Query Localization Head with Deep Supervision and Hungarian Matching.

Per Section 7.7:
- 30 learnable event queries
- Transformer decoder with cross-attention to temporal encoder representations
- Deep supervision across decoder layers
- Predictions per query:
  * Event classification (foreground vs background)
  * Temporal boundaries [center, duration] in normalized range [0, 1]
- Exact Hungarian Matching Cost weights: 5 (L1) / 2 (tIoU) / 1 (classification)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def segment_tiou(seg1: torch.Tensor, seg2: torch.Tensor) -> torch.Tensor:
    """Computes pairwise temporal IoU between seg1 [N, 2] and seg2 [M, 2].
    Segments are in [start, end] format.
    """
    s1, e1 = seg1[:, 0].unsqueeze(1), seg1[:, 1].unsqueeze(1)
    s2, e2 = seg2[:, 0].unsqueeze(0), seg2[:, 1].unsqueeze(0)

    inter = torch.clamp(torch.min(e1, e2) - torch.max(s1, s2), min=0.0)
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / torch.clamp(union, min=1e-6)


class HungarianMatcher1D(nn.Module):
    def __init__(self, cost_l1: float = 5.0, cost_tiou: float = 2.0, cost_class: float = 1.0):
        super().__init__()
        self.cost_l1 = cost_l1
        self.cost_tiou = cost_tiou
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, pred_logits: torch.Tensor, pred_segs: torch.Tensor, targets: list):
        """Bipartite matching between predicted 30 event queries and ground-truth intervals.
        
        Args:
            pred_logits: Tensor of shape [B, 30, 1]
            pred_segs: Tensor of shape [B, 30, 2] in [s, e] format in [0, 1]
            targets: list of B dicts, each with "segments" [N_gt, 2] and "labels" [N_gt]
        Returns:
            matches: list of tuples (pred_indices, gt_indices)
        """
        B, num_queries, _ = pred_logits.shape
        matches = []

        for b in range(B):
            tgt_segs = targets[b]["segments"].to(pred_segs.device) # [N_gt, 2]
            if len(tgt_segs) == 0:
                matches.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue

            # Classification cost: BCE / focal probability
            p_prob = torch.sigmoid(pred_logits[b]).squeeze(-1) # [30]
            # Cost for matching positive target is -prob
            cost_cls = -p_prob.unsqueeze(1).repeat(1, len(tgt_segs)) # [30, N_gt]

            # L1 segment cost: |s - s_gt| + |e - e_gt|
            b_segs = pred_segs[b] # [30, 2]
            cost_l1 = torch.cdist(b_segs, tgt_segs, p=1) # [30, N_gt]

            # tIoU cost: -tIoU
            tiou = segment_tiou(b_segs, tgt_segs) # [30, N_gt]
            cost_tiou = -tiou

            # Weighted sum: 5 (L1) / 2 (tIoU) / 1 (class)
            C = self.cost_l1 * cost_l1 + self.cost_tiou * cost_tiou + self.cost_class * cost_cls
            C_cpu = C.detach().cpu().numpy()

            pred_idx, gt_idx = linear_sum_assignment(C_cpu)
            matches.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(gt_idx, dtype=torch.int64)
            ))

        return matches


class EventQueryLocalizationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_queries: int = 30,
        num_decoder_layers: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # 30 Learnable event queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Transformer decoder layers with cross-attention to video sequence
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Prediction heads with Deep Supervision
        self.class_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_decoder_layers)
        ])
        self.segment_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Sigmoid() # Center and duration in [0, 1]
            ) for _ in range(num_decoder_layers)
        ])

        self.matcher = HungarianMatcher1D(cost_l1=5.0, cost_tiou=2.0, cost_class=1.0)

    def _convert_cd_to_se(self, cd: torch.Tensor) -> torch.Tensor:
        """Converts [center, duration] to [start, end] format, clamped in [0, 1]."""
        center = cd[..., 0]
        duration = cd[..., 1]
        start = torch.clamp(center - duration / 2.0, min=0.0, max=1.0)
        end = torch.clamp(center + duration / 2.0, min=0.0, max=1.0)
        return torch.stack([start, end], dim=-1)

    def forward(self, encoder_out: torch.Tensor):
        """Forward pass for 30 event queries over encoder sequence.
        
        Args:
            encoder_out: Tensor of shape [B, T, 512] or [T, 512]
        Returns:
            dict containing:
              - pred_logits: [B, 30, 1] (final layer)
              - pred_segments: [B, 30, 2] (final layer [start, end])
              - query_features: [B, 30, 512] (final layer representation for captioning)
              - aux_outputs: list of dicts for deep supervision across prior layers
        """
        is_unbatched = (encoder_out.ndim == 2)
        if is_unbatched:
            encoder_out = encoder_out.unsqueeze(0)

        B, T, _ = encoder_out.shape

        # Initialize queries: [B, 30, 512]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Decoder pass with intermediate states for deep supervision
        intermediate_states = []
        h = queries
        for mod in self.decoder.layers:
            h = mod(h, encoder_out)
            intermediate_states.append(h)

        aux_outputs = []
        for l_idx in range(self.num_decoder_layers):
            state = intermediate_states[l_idx]
            logits = self.class_heads[l_idx](state) # [B, 30, 1]
            cd = self.segment_heads[l_idx](state)   # [B, 30, 2]
            segs = self._convert_cd_to_se(cd)       # [B, 30, 2]
            aux_outputs.append({
                "pred_logits": logits,
                "pred_segments": segs,
                "query_features": state
            })

        # Final predictions from the top layer
        final_out = aux_outputs[-1]
        result = {
            "pred_logits": final_out["pred_logits"],
            "pred_segments": final_out["pred_segments"],
            "query_features": final_out["query_features"],
            "aux_outputs": aux_outputs[:-1] # Deep supervision auxiliary layers
        }

        if is_unbatched:
            result["pred_logits"] = result["pred_logits"].squeeze(0)
            result["pred_segments"] = result["pred_segments"].squeeze(0)
            result["query_features"] = result["query_features"].squeeze(0)

        return result

    def compute_loss(self, outputs: dict, targets: list) -> dict:
        """Computes Hungarian matched loss with deep supervision."""
        # Main layer loss
        loss_dict = self._compute_single_layer_loss(
            outputs["pred_logits"],
            outputs["pred_segments"],
            targets
        )

        # Auxiliary deep supervision losses
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_loss = self._compute_single_layer_loss(
                aux["pred_logits"],
                aux["pred_segments"],
                targets
            )
            for k, v in aux_loss.items():
                loss_dict[f"{k}_aux_{i}"] = v

        total_loss = sum(v for k, v in loss_dict.items() if "loss" in k)
        loss_dict["total_loss"] = total_loss
        return loss_dict

    def _compute_single_layer_loss(self, pred_logits, pred_segs, targets):
        if pred_logits.ndim == 2:
            pred_logits = pred_logits.unsqueeze(0)
            pred_segs = pred_segs.unsqueeze(0)

        B = pred_logits.size(0)
        device = pred_logits.device
        matches = self.matcher(pred_logits, pred_segs, targets)

        loss_cls = 0.0
        loss_l1 = 0.0
        loss_tiou = 0.0
        num_pos_total = 0

        for b in range(B):
            pred_idx, gt_idx = matches[b]
            tgt_segs = targets[b]["segments"].to(device)
            N_gt = len(tgt_segs)

            # Target class binary vector for 30 queries
            target_cls = torch.zeros(self.num_queries, device=device)
            if len(pred_idx) > 0:
                target_cls[pred_idx] = 1.0
                num_pos_total += len(pred_idx)

                # Segment regression loss on matched pairs
                matched_preds = pred_segs[b, pred_idx]
                matched_tgts = tgt_segs[gt_idx]

                loss_l1 += F.l1_loss(matched_preds, matched_tgts, reduction="sum")
                # Negative tIoU loss: 1 - tIoU
                tiou = torch.diag(segment_tiou(matched_preds, matched_tgts))
                loss_tiou += (1.0 - tiou).sum()

            # Classification loss across all 30 queries
            loss_cls += F.binary_cross_entropy_with_logits(
                pred_logits[b].squeeze(-1),
                target_cls,
                reduction="sum"
            )

        norm = max(1, num_pos_total)
        return {
            "loss_cls": loss_cls / (B * self.num_queries),
            "loss_l1": (loss_l1 / norm) * 5.0,     # Hungarian weight 5
            "loss_tiou": (loss_tiou / norm) * 2.0  # Hungarian weight 2
        }
