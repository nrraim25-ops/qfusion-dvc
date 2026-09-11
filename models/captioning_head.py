"""Captioning Head with GloVe embeddings, Autoregressive Decoder, and Beam Search.

Per Section 7.9:
- Fuses event query features [512] with soft-pooled context vectors [512] -> [512]
- Word embeddings initialized from 300-dim GloVe vectors projected to hidden_dim (512)
- Autoregressive generation with teacher forcing for training
- Beam search with width 5 for evaluation / inference
"""

import math
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptioningDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 512,
        glove_dim: int = 300,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
        unk_idx: int = 3,
        beam_width: int = 5,
        max_seq_len: int = 25
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.glove_dim = glove_dim
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx
        self.beam_width = beam_width
        self.max_seq_len = max_seq_len

        # Event representation fusion: query_feats (512) + context_vec (512) -> 512
        self.event_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Word embedding layer (300d GloVe projection)
        self.word_embed = nn.Embedding(vocab_size, glove_dim, padding_idx=pad_idx)
        self.embed_proj = nn.Linear(glove_dim, embed_dim)

        # Autoregressive GRU Decoder
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True
        )

        # Output projection to vocabulary logits
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def load_glove_embeddings(self, glove_matrix: torch.Tensor):
        """Initializes the embedding table with pretrained GloVe 300d vectors."""
        with torch.no_grad():
            self.word_embed.weight.copy_(glove_matrix)

    def forward(
        self,
        query_feats: torch.Tensor,
        context_vec: torch.Tensor,
        caption_tokens: torch.Tensor = None
    ):
        """Forward pass for captioning decoder.
        
        Args:
            query_feats: Tensor of shape [B, num_queries, 512]
            context_vec: Tensor of shape [B, num_queries, 512]
            caption_tokens: Tensor of shape [B, num_queries, seq_len] (ground truth for teacher forcing)
        Returns:
            If caption_tokens is provided:
                logits: Tensor of shape [B, num_queries, seq_len, vocab_size]
            Else (inference):
                generated_tokens: Tensor of shape [B, num_queries, max_seq_len] (beam search width 5)
        """
        B, num_queries, D = query_feats.shape

        # Fuse query and soft-pooled context
        fused_input = torch.cat([query_feats, context_vec], dim=-1) # [B, num_queries, 1024]
        event_reps = self.event_fusion(fused_input)                 # [B, num_queries, 512]

        # Flatten B and num_queries for decoder processing: [B * num_queries, 512]
        event_reps_flat = event_reps.view(B * num_queries, D)

        if caption_tokens is not None:
            # Training with teacher forcing
            seq_len = caption_tokens.size(-1)
            tokens_flat = caption_tokens.view(B * num_queries, seq_len)

            # Word embeddings: [B*num_queries, seq_len, 300] -> [B*num_queries, seq_len, 512]
            embeddings = self.embed_proj(self.word_embed(tokens_flat))

            # Initial hidden state conditioned on event representation
            h_0 = event_reps_flat.unsqueeze(0) # [1, B*num_queries, 512]

            rnn_out, _ = self.rnn(embeddings, h_0) # [B*num_queries, seq_len, 512]
            logits = self.fc_out(rnn_out)          # [B*num_queries, seq_len, vocab_size]

            return logits.view(B, num_queries, seq_len, self.vocab_size)

        else:
            # Inference: Beam Search Decoding (Width 5)
            generated = self.beam_search(event_reps_flat, beam_width=self.beam_width)
            return generated.view(B, num_queries, -1)

    def beam_search(self, event_reps: torch.Tensor, beam_width: int = 5) -> torch.Tensor:
        """Runs beam search decoding with width 5 over event representations."""
        N = event_reps.size(0)
        device = event_reps.device

        # All generated sequences: [N, max_seq_len]
        all_sequences = []

        for i in range(N):
            h = event_reps[i : i + 1].unsqueeze(0) # [1, 1, 512]
            # Beams: list of (log_prob, [token_ids], h_state)
            beams = [(0.0, [self.bos_idx], h)]

            for step in range(self.max_seq_len - 1):
                new_beams = []
                all_done = True

                for score, seq, h_state in beams:
                    if seq[-1] == self.eos_idx:
                        new_beams.append((score, seq, h_state))
                        continue

                    all_done = False
                    curr_tok = torch.tensor([[seq[-1]]], device=device)
                    curr_emb = self.embed_proj(self.word_embed(curr_tok)) # [1, 1, 512]

                    out, next_h = self.rnn(curr_emb, h_state)
                    log_probs = F.log_softmax(self.fc_out(out.squeeze(1)), dim=-1).squeeze(0) # [vocab_size]

                    topk_scores, topk_indices = torch.topk(log_probs, k=beam_width)
                    for k in range(beam_width):
                        tok = topk_indices[k].item()
                        new_score = score + topk_scores[k].item()
                        new_beams.append((new_score, seq + [tok], next_h))

                if all_done:
                    break

                # Prune to top beam_width beams
                new_beams.sort(key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)
                beams = new_beams[:beam_width]

            best_seq = beams[0][1]
            # Pad to max_seq_len
            if len(best_seq) < self.max_seq_len:
                best_seq = best_seq + [self.pad_idx] * (self.max_seq_len - len(best_seq))
            all_sequences.append(best_seq[:self.max_seq_len])

        return torch.tensor(all_sequences, device=device, dtype=torch.long)
