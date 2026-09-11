"""Unified Dense Video Captioning (DVC) End-to-End Architecture.

Integrates:
- Video and Audio Inputs (ViViT and AST backbones, with TSP pretraining)
- Modular Fusion: 'classical' | 'classical_matched' | 'quantum'
- Deformable + Sparse-Attention Temporal Encoder
- 30-Query Event Localization Head (Deep Supervision, Hungarian loss weights 5/2/1)
- Differentiable Sigmoid Context Mask
- Captioning Decoder (GloVe 300d embeddings, beam width 5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion_classical import ClassicalFusion
from models.fusion_classical_matched import ClassicalMatchedFusion
from models.fusion_quantum import QuantumFusion
from models.encoder_deformable import DeformableSparseEncoder
from models.localization_head import EventQueryLocalizationHead
from models.context_mask import DifferentiableContextMask
from models.captioning_head import CaptioningDecoder


class DVCModel(nn.Module):
    def __init__(
        self,
        fusion_type: str = "classical",
        embed_dim: int = 512,
        num_queries: int = 30,
        vocab_size: int = 5000,
        glove_dim: int = 300,
        beam_width: int = 5,
        max_seq_len: int = 25
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.vocab_size = vocab_size

        # 1. Fusion Module
        if fusion_type == "classical":
            self.fusion = ClassicalFusion(video_dim=768, audio_dim=768, fused_dim=embed_dim)
        elif fusion_type == "classical_matched":
            self.fusion = ClassicalMatchedFusion(video_dim=768, audio_dim=768, fused_dim=embed_dim)
        elif fusion_type == "quantum":
            self.fusion = QuantumFusion(video_dim=768, audio_dim=768, fused_dim=embed_dim, num_qubits=6, num_layers=4)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from 'classical', 'classical_matched', 'quantum'.")

        # 2. Deformable + Sparse-Attention Encoder
        self.encoder = DeformableSparseEncoder(
            embed_dim=embed_dim,
            num_layers=4,
            num_heads=4,
            num_points=4,
            prune_ratio=0.5
        )

        # 3. 30-Query Localization Head with Deep Supervision
        self.localization_head = EventQueryLocalizationHead(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_decoder_layers=3,
            num_heads=8
        )

        # 4. Differentiable Context Mask
        self.context_mask = DifferentiableContextMask(
            embed_dim=embed_dim,
            temperature=10.0
        )

        # 5. Captioning Decoder
        self.captioning_head = CaptioningDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            glove_dim=glove_dim,
            beam_width=beam_width,
            max_seq_len=max_seq_len
        )

    def forward(
        self,
        video_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        caption_tokens: torch.Tensor = None,
        generate_captions: bool = False
    ) -> dict:
        """Full forward pass for DVC.
        
        Args:
            video_feats: Tensor of shape [B, T, 768] or [T, 768]
            audio_feats: Tensor of shape [B, T, 768] or [T, 768]
            caption_tokens: Tensor of shape [B, 30, seq_len] (ground truth for training teacher forcing)
        Returns:
            dict containing:
              - pred_logits: [B, 30, 1]
              - pred_segments: [B, 30, 2]
              - query_features: [B, 30, 512]
              - aux_outputs: list of auxiliary deep supervision predictions
              - context_vectors: [B, 30, 512]
              - fused_feats: [B, T, 512]
              - encoder_out: [B, T, 512]
              - caption_logits (training) or generated_tokens (inference): [B, 30, seq_len, vocab_size] / [B, 30, max_seq_len]
        """
        is_unbatched = (video_feats.ndim == 2)
        if is_unbatched:
            video_feats = video_feats.unsqueeze(0)
            audio_feats = audio_feats.unsqueeze(0)
            if caption_tokens is not None and caption_tokens.ndim == 2:
                caption_tokens = caption_tokens.unsqueeze(0)

        # 1. Multimodal Fusion
        fused_feats = self.fusion(video_feats, audio_feats) # [B, T, 512]

        # 2. Deformable + Sparse Attention Encoder
        encoder_out, saliency_scores, keep_indices = self.encoder(fused_feats) # [B, T, 512]

        # 3. Event Query Localization Head
        loc_outputs = self.localization_head(encoder_out)
        pred_logits = loc_outputs["pred_logits"]       # [B, 30, 1]
        pred_segments = loc_outputs["pred_segments"]   # [B, 30, 2]
        query_features = loc_outputs["query_features"] # [B, 30, 512]

        # 4. Differentiable Sigmoid Context Mask
        context_vectors, mask_weights = self.context_mask(encoder_out, pred_segments) # [B, 30, 512]

        # 5. Captioning Decoder
        if caption_tokens is not None:
            caption_out = self.captioning_head(
                query_features,
                context_vectors,
                caption_tokens=caption_tokens
            ) # [B, 30, seq_len, vocab_size]
        elif generate_captions:
            caption_out = self.captioning_head(
                query_features,
                context_vectors,
                caption_tokens=None
            ) # [B, 30, max_seq_len] (beam search during evaluation)
        else:
            caption_out = None

        outputs = {
            "pred_logits": pred_logits,
            "pred_segments": pred_segments,
            "query_features": query_features,
            "aux_outputs": loc_outputs.get("aux_outputs", []),
            "context_vectors": context_vectors,
            "fused_feats": fused_feats,
            "encoder_out": encoder_out,
            "caption_out": caption_out,
            "saliency_scores": saliency_scores
        }

        if is_unbatched:
            for k in ["pred_logits", "pred_segments", "query_features", "context_vectors", "fused_feats", "encoder_out", "caption_out"]:
                outputs[k] = outputs[k].squeeze(0)

        return outputs

    def compute_loss(
        self,
        outputs: dict,
        targets: list,
        caption_targets: torch.Tensor = None
    ) -> dict:
        """Computes combined DVC loss: Localization (Hungarian) + Captioning (CrossEntropy)."""
        # 1. Localization loss
        loc_loss_dict = self.localization_head.compute_loss(outputs, targets)

        # 2. Captioning loss (on matched events)
        total_cap_loss = torch.tensor(0.0, device=outputs["pred_logits"].device)
        if caption_targets is not None:
            # Match queries to ground truth to compute caption loss on positive events
            matches = self.localization_head.matcher(outputs["pred_logits"], outputs["pred_segments"], targets)
            num_matched = 0
            all_logits = []
            all_targets = []

            for b in range(len(targets)):
                pred_idx, gt_idx = matches[b]
                if len(pred_idx) > 0:
                    q_feats = outputs["query_features"][b, pred_idx].unsqueeze(0)
                    c_vecs = outputs["context_vectors"][b, pred_idx].unsqueeze(0)
                    gt_toks = caption_targets[b, gt_idx].unsqueeze(0)

                    inp_toks = gt_toks[:, :, :-1]
                    tgt_toks = gt_toks[:, :, 1:]

                    cap_logits = self.captioning_head(q_feats, c_vecs, caption_tokens=inp_toks)
                    all_logits.append(cap_logits.view(-1, self.vocab_size))
                    all_targets.append(tgt_toks.reshape(-1))
                    num_matched += len(pred_idx)

            if num_matched > 0 and len(all_logits) > 0:
                cat_logits = torch.cat(all_logits, dim=0)
                cat_targets = torch.cat(all_targets, dim=0)
                total_cap_loss = F.cross_entropy(cat_logits, cat_targets, ignore_index=0)

        loc_loss_dict["loss_caption"] = total_cap_loss
        loc_loss_dict["total_loss"] = loc_loss_dict["total_loss"] + total_cap_loss * 2.0
        return loc_loss_dict
