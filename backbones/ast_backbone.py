"""AST (Audio Spectrogram Transformer) backbone for audio snippet feature extraction."""

import os
import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig, ASTFeatureExtractor


class ASTBackbone(nn.Module):
    def __init__(self, pretrained_model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593", pretrained: bool = True):
        super().__init__()
        try:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model_name)
        except Exception:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model_name, local_files_only=True)

        if pretrained:
            try:
                self.ast = ASTModel.from_pretrained(pretrained_model_name, use_safetensors=True)
            except Exception:
                self.ast = ASTModel.from_pretrained(pretrained_model_name, use_safetensors=True, local_files_only=True)
        else:
            try:
                config = ASTConfig.from_pretrained(pretrained_model_name)
            except Exception:
                config = ASTConfig.from_pretrained(pretrained_model_name, local_files_only=True)
            self.ast = ASTModel(config)

        self.ast.gradient_checkpointing_enable()
        self.embed_dim = self.ast.config.hidden_size # 768

    def forward(self, audio_snippets: torch.Tensor, max_chunk_size: int = 8) -> torch.Tensor:
        """Forward pass for audio snippets.
        
        Args:
            audio_snippets: Tensor of shape [N, 32000] (can be on CPU or GPU)
            max_chunk_size: Micro-chunk size for feature extraction
        Returns:
            audio_feats: Tensor of shape [N, 768] (float32)
        """
        target_device = next(self.ast.parameters()).device
        N = audio_snippets.size(0)

        def _forward_chunk(chunk):
            chunk_np = chunk.detach().cpu().numpy()
            inputs = self.feature_extractor(
                chunk_np,
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_values = inputs.input_values.to(target_device) # [B, 1024, 128]

            outputs = self.ast(input_values=input_values)
            if outputs.pooler_output is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            return feats.float().cpu()

        if N <= max_chunk_size:
            return _forward_chunk(audio_snippets).to(target_device)

        chunks = []
        for i in range(0, N, max_chunk_size):
            chunk = audio_snippets[i : i + max_chunk_size]
            chunks.append(_forward_chunk(chunk))
        return torch.cat(chunks, dim=0).to(target_device)
