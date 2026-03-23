"""Depth Transformer for OmniVoxtral.

The Depth Transformer predicts inter-codebook dependencies at each timestep.
Given the Temporal Transformer's hidden state, it autoregressively generates
codebook tokens q1 → q2 → ... → q8.

Architecture (from Moshi paper §4):
- 6 layers, 16 heads, dim 1024
- Causal attention within the codebook sequence
- Input: projected temporal hidden state + codebook token embeddings
- Output: logits over codebook vocabulary for each position

This is separate from the Temporal Transformer to keep per-timestep
computation bounded — the Depth Transformer runs once per timestep,
not once per token in the full sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class DepthTransformerConfig:
    """Configuration for the Depth Transformer."""

    num_layers: int = 6
    dim: int = 1024
    num_heads: int = 16
    num_codebooks: int = 8
    codebook_size: int = 2048  # Mimi codebook cardinality
    temporal_dim: int = 4096  # Temporal Transformer hidden dim (Mistral)
    dropout: float = 0.0
    max_seq_len: int = 16  # num_codebooks + text token + padding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from Mistral/Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class DepthAttention(nn.Module):
    """Multi-head causal self-attention for the Depth Transformer."""

    def __init__(self, config: DepthTransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.dim = config.dim

        self.q_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        return self.o_proj(attn_output)


class DepthFeedForward(nn.Module):
    """SwiGLU feed-forward network (matching Mistral's architecture)."""

    def __init__(self, config: DepthTransformerConfig):
        super().__init__()
        hidden_dim = int(config.dim * 8 / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DepthTransformerLayer(nn.Module):
    """Single layer of the Depth Transformer: attention + feed-forward with pre-norm."""

    def __init__(self, config: DepthTransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim)
        self.attn = DepthAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = DepthFeedForward(config)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class DepthTransformer(nn.Module):
    """Depth Transformer: predicts codebook tokens q1→q2→...→q8 at each timestep.

    At each timestep t, the Temporal Transformer produces a hidden state h_t.
    The Depth Transformer takes h_t and autoregressively generates codebook tokens:

        Input:  [h_t_projected, emb(q1), emb(q2), ..., emb(q7)]
        Output: [logits_q1,     logits_q2, logits_q3, ..., logits_q8]

    The first position always receives the projected temporal hidden state.
    Subsequent positions receive the embedding of the previously generated codebook token.
    """

    def __init__(self, config: DepthTransformerConfig):
        super().__init__()
        self.config = config

        # Project temporal hidden state down to depth dimension
        self.input_proj = nn.Linear(config.temporal_dim, config.dim, bias=False)

        # Per-codebook token embeddings (each codebook has its own embedding table)
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_size, config.dim)
            for _ in range(config.num_codebooks)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            DepthTransformerLayer(config) for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.dim)

        # Per-codebook output heads (each predicts over its own vocabulary)
        self.output_heads = nn.ModuleList([
            nn.Linear(config.dim, config.codebook_size, bias=False)
            for _ in range(config.num_codebooks)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small std for stable training from scratch."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        temporal_hidden: torch.Tensor,
        codebook_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training (teacher forcing).

        Args:
            temporal_hidden: Hidden state from Temporal Transformer.
                Shape: (batch_size, dim_temporal) or (batch_size, num_timesteps, dim_temporal)
            codebook_tokens: Ground truth codebook tokens for teacher forcing.
                Shape: (batch_size, num_codebooks) or (batch_size, num_timesteps, num_codebooks)
                Values in [0, codebook_size). If None, returns logits for first codebook only.

        Returns:
            logits: Shape (batch_size, num_codebooks, codebook_size) or
                    (batch_size, num_timesteps, num_codebooks, codebook_size)
        """
        # Handle both single-timestep and batched-timestep inputs
        if temporal_hidden.dim() == 2:
            # Single timestep: (batch, dim) → (batch, 1, dim) → process → squeeze
            return self._forward_single(temporal_hidden, codebook_tokens)
        else:
            # Multiple timesteps: process each independently
            # (batch, T, dim) → reshape to (batch*T, dim) → process → reshape back
            batch_size, num_timesteps, _ = temporal_hidden.shape
            h_flat = temporal_hidden.reshape(batch_size * num_timesteps, -1)

            cb_flat = None
            if codebook_tokens is not None:
                cb_flat = codebook_tokens.reshape(batch_size * num_timesteps, -1)

            logits_flat = self._forward_single(h_flat, cb_flat)
            # logits_flat: (batch*T, num_codebooks, codebook_size)
            return logits_flat.reshape(
                batch_size, num_timesteps, self.config.num_codebooks, self.config.codebook_size
            )

    def _forward_single(
        self,
        temporal_hidden: torch.Tensor,
        codebook_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process a single timestep (or batch of single timesteps).

        Args:
            temporal_hidden: (batch_size, dim_temporal)
            codebook_tokens: (batch_size, num_codebooks) or None

        Returns:
            logits: (batch_size, num_codebooks, codebook_size)
        """
        batch_size = temporal_hidden.size(0)
        num_q = self.config.num_codebooks

        # Project temporal hidden state: (batch, temporal_dim) → (batch, depth_dim)
        h0 = self.input_proj(temporal_hidden)  # (batch, dim)

        # Build input sequence for depth transformer
        # Position 0: projected temporal hidden
        # Position i (1..num_q-1): embedding of codebook token q_i
        seq = [h0.unsqueeze(1)]  # [(batch, 1, dim)]

        if codebook_tokens is not None:
            # Teacher forcing: use ground truth tokens
            # Optional: randomly corrupt tokens to mitigate exposure bias
            if self.training and hasattr(self, '_mask_rate') and self._mask_rate > 0:
                mask = torch.rand(batch_size, num_q - 1, device=codebook_tokens.device) < self._mask_rate
                random_tokens = torch.randint(0, self.config.codebook_size, (batch_size, num_q - 1), device=codebook_tokens.device)
                cb_input = torch.where(mask, random_tokens, codebook_tokens[:, :num_q - 1])
            else:
                cb_input = codebook_tokens[:, :num_q - 1]
            for i in range(num_q - 1):
                tok_emb = self.codebook_embeddings[i](cb_input[:, i])
                seq.append(tok_emb.unsqueeze(1))

        # Concatenate: (batch, num_positions, dim)
        x = torch.cat(seq, dim=1)  # (batch, 1..num_q, dim)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Compute logits for each codebook position
        # Position 0 predicts q1, position 1 predicts q2, etc.
        all_logits = []
        for i in range(min(x.size(1), num_q)):
            logits_i = self.output_heads[i](x[:, i, :])  # (batch, codebook_size)
            all_logits.append(logits_i)

        # If we only have position 0 (no teacher forcing), pad remaining
        while len(all_logits) < num_q:
            all_logits.append(torch.zeros(
                batch_size, self.config.codebook_size,
                device=x.device, dtype=x.dtype
            ))

        return torch.stack(all_logits, dim=1)  # (batch, num_q, codebook_size)

    @torch.no_grad()
    def generate(
        self,
        temporal_hidden: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation of codebook tokens.

        Args:
            temporal_hidden: (batch_size, dim_temporal)
            temperature: Sampling temperature
            top_k: Top-k filtering (None = no filtering)

        Returns:
            tokens: (batch_size, num_codebooks) — generated codebook token IDs
        """
        batch_size = temporal_hidden.size(0)
        num_q = self.config.num_codebooks

        # Start with projected temporal hidden
        h0 = self.input_proj(temporal_hidden)
        generated_tokens = []

        # Build sequence incrementally
        seq = [h0.unsqueeze(1)]  # [(batch, 1, dim)]

        for i in range(num_q):
            # Concatenate current sequence
            x = torch.cat(seq, dim=1)

            # Forward through all layers
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)

            # Get logits for current position
            logits = self.output_heads[i](x[:, -1, :])  # (batch, codebook_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_vals, _ = logits.topk(top_k, dim=-1)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Sample
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (batch,)
            generated_tokens.append(token)

            # Embed and append for next position (except last)
            if i < num_q - 1:
                tok_emb = self.codebook_embeddings[i](token)
                seq.append(tok_emb.unsqueeze(1))

        return torch.stack(generated_tokens, dim=1)  # (batch, num_q)
