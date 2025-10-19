"""
Minimal GPT (character-level or word-level) implemented with PyTorch.
This file defines a small Transformer-based next-token predictor.
It's intentionally compact and educational, not optimized for production.
"""
import math
import torch
import torch.nn as nn


class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd, dropout=dropout, activation='gelu')
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"
        tok_emb = self.tok_emb(idx)  # (B, T, C)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)
        # transformer encoder layers expect (T, B, C)
        x = x.transpose(0, 1)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(0, 1)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits

    def generate(self, idx, max_new_tokens, eos_token=None):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            T = idx.size(1)
            if T > self.block_size:
                idx = idx[:, -self.block_size:]
            logits = self.forward(idx)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_token is not None:
                if (next_token == eos_token).any():
                    break
        return idx
