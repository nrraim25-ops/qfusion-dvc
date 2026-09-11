"""ViViT (Video Vision Transformer) backbone for video snippet feature extraction."""

import os
import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig
from pathlib import Path

class ViViTBackbone(nn.Module):
    def __init__(self, pretrained_model_name: str = "google/vivit-b-16x2-kinetics400", pretrained: bool = True):
        super().__init__()
        if pretrained:
            try:
                self.vivit = VivitModel.from_pretrained(pretrained_model_name, use_safetensors=True)
            except Exception:
                self.vivit = VivitModel.from_pretrained(pretrained_model_name, use_safetensors=True, local_files_only=True)
        else:
            try:
                config = VivitConfig.from_pretrained(pretrained_model_name)
            except Exception:
                config = VivitConfig.from_pretrained(pretrained_model_name, local_files_only=True)
            self.vivit = VivitModel(config)
        
        self.vivit.gradient_checkpointing_enable()
        self.embed_dim = self.vivit.config.hidden_size # 768

    def forward(self, snippet_frames: torch.Tensor, max_chunk_size: int = 2) -> torch.Tensor:
        """Forward pass for video snippets with automatic micro-chunking and bfloat16 autocast.
        
        Args:
            snippet_frames: Tensor of shape [N, 32, 3, 224, 224] (can be on CPU or GPU)
            max_chunk_size: Max snippets per GPU forward sub-pass (default 2)
        Returns:
            video_feats: Tensor of shape [N, 768] (float32)
        """
        N = snippet_frames.size(0)
        target_device = next(self.vivit.parameters()).device
        device_type = target_device.type

        def _forward_chunk(chunk):
            chunk_gpu = chunk.to(target_device)
            if device_type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.vivit(pixel_values=chunk_gpu)
            else:
                outputs = self.vivit(pixel_values=chunk_gpu)
            feats = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
            return feats.float().cpu()

        if N <= max_chunk_size:
            return _forward_chunk(snippet_frames).to(target_device)

        chunks = []
        for i in range(0, N, max_chunk_size):
            chunk = snippet_frames[i : i + max_chunk_size]
            chunks.append(_forward_chunk(chunk))
        return torch.cat(chunks, dim=0).to(target_device)
