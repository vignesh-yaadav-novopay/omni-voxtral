"""OmniVoxtral: Dual-transformer architecture for multilingual duplex speech LM.

Combines:
- Temporal Transformer (Mistral 7B): models time dependencies across the full context
- Depth Transformer (113M): models inter-codebook dependencies at each timestep

The Temporal Transformer processes the interleaved token sequence and produces
hidden states. The Depth Transformer takes each hidden state at audio positions
and predicts the codebook tokens q1→q2→...→q8.

Architecture follows Moshi (arXiv:2410.00037) §4, adapted for multilingual support.
"""

import logging

import torch
import torch.nn as nn
import transformers as tr

from voxtral.model.depth_transformer import DepthTransformer, DepthTransformerConfig
from voxtral.model.language_adapters import (
    activate_adapter,
    create_language_adapters,
    get_adapter_info,
)

logger = logging.getLogger(__name__)


class OmniVoxtralConfig:
    """Configuration for the full OmniVoxtral model."""

    def __init__(
        self,
        # Temporal Transformer (Mistral)
        temporal_pretrained_path: str = "mistralai/Mistral-7B-v0.3",
        temporal_kwargs: dict | None = None,
        # Total vocab: text_vocab_size + num_codebooks * codebook_size
        # 65536 + 8 * 2048 = 81920, rounded up for alignment
        new_vocab_size: int = 81920,
        prune_layers: int | None = None,
        lora_rank: int | None = None,
        # Depth Transformer
        depth_num_layers: int = 6,
        depth_dim: int = 1024,
        depth_num_heads: int = 16,
        depth_dropout: float = 0.0,
        # Audio config
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        text_vocab_size: int = 65536,
        text_hz: int = 5,
        # Dual-stream config
        dual_stream: bool = False,
        # Language adapter config
        language_adapters: bool = False,
        adapter_rank: int = 8,
        adapter_alpha: int = 16,
    ):
        self.temporal_pretrained_path = temporal_pretrained_path
        self.temporal_kwargs = temporal_kwargs or {}
        self.new_vocab_size = new_vocab_size
        self.prune_layers = prune_layers
        self.lora_rank = lora_rank

        self.depth_num_layers = depth_num_layers
        self.depth_dim = depth_dim
        self.depth_num_heads = depth_num_heads
        self.depth_dropout = depth_dropout

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.text_vocab_size = text_vocab_size
        self.text_hz = text_hz
        self.dual_stream = dual_stream

        self.language_adapters = language_adapters
        self.adapter_rank = adapter_rank
        self.adapter_alpha = adapter_alpha

        # Derived
        self.text_to_audio_factor = int(12.5 * num_codebooks / text_hz)  # 20
        self.stream_stride = 1 + self.text_to_audio_factor  # 21 tokens per stream
        # In dual-stream mode, each window has user + model = 42 tokens
        # In single-stream mode, stride = 21 (backward compatible)
        self.stride = 2 * self.stream_stride if dual_stream else self.stream_stride


class OmniVoxtral(nn.Module):
    """OmniVoxtral dual-transformer model.

    Single-stream layout per 200ms window (stride=21):
        [text, q1_f1, q2_f1, ..., q8_f1, q1_f2, ..., q8_f2, q1_f3, ..., q4_f3]

    Dual-stream layout per 200ms window (stride=42):
        [user_text, user_q1_f1...user_q8_f2..., model_text, model_q1_f1...model_q8_f2...]
        i.e. [user_stream(21), model_stream(21)]

    In dual-stream mode:
        - The Depth Transformer operates on the MODEL stream's hidden states
        - The model stream text position is at offset stream_stride (21) within each window
        - Audio codebook targets come from the model stream only

    The Temporal Transformer predicts all tokens in the sequence.
    The Depth Transformer predicts codebook tokens q1->q2->...->q8 given hidden states.
    """

    def __init__(self, config: OmniVoxtralConfig):
        super().__init__()
        self.config = config

        # --- Temporal Transformer (Mistral) ---
        # NOTE: We do NOT use output_hidden_states=True here. That would store ALL 32
        # layer outputs (~2GB wasted memory). Instead, forward() calls model.model()
        # to get just the last hidden state, then applies lm_head manually.
        self.temporal = tr.MistralForCausalLM.from_pretrained(
            config.temporal_pretrained_path,
            use_cache=False,
            eos_token_id=99999,
            output_hidden_states=False,
            low_cpu_mem_usage=True,     # shard-by-shard loading (1x RAM, not 2x)
            torch_dtype=torch.bfloat16,  # 7GB for 7B model instead of 14GB
            **config.temporal_kwargs,
        )
        self.temporal.resize_token_embeddings(config.new_vocab_size)

        # Apply layer pruning if configured
        if config.prune_layers is not None:
            layers_to_remove = list(
                range(
                    config.prune_layers - 1,
                    len(self.temporal.model.layers),
                    config.prune_layers,
                )
            )
            for layer_idx in sorted(layers_to_remove, reverse=True):
                del self.temporal.model.layers[layer_idx]
            self.temporal.config.num_hidden_layers = len(self.temporal.model.layers)

        # Get temporal hidden dimension
        self.temporal_dim = self.temporal.config.hidden_size

        # --- Language Adapters (applied before Depth, since they modify temporal) ---
        if config.language_adapters:
            self.temporal = create_language_adapters(
                self.temporal,
                adapter_rank=config.adapter_rank,
                adapter_alpha=config.adapter_alpha,
            )
            logger.info(f"Language adapters created: {get_adapter_info(self.temporal)}")

        # --- Depth Transformer ---
        depth_config = DepthTransformerConfig(
            num_layers=config.depth_num_layers,
            dim=config.depth_dim,
            num_heads=config.depth_num_heads,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            temporal_dim=self.temporal_dim,
            dropout=config.depth_dropout,
        )
        self.depth = DepthTransformer(depth_config)

    def _temporal_forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run temporal transformer, returning (hidden_states, logits).

        Handles two cases:
        - Standard MistralForCausalLM: calls model.model() + lm_head() directly,
          avoiding materializing all 32 intermediate hidden states (~2GB savings).
        - PEFT-wrapped model: calls through PEFT's forward with output_hidden_states=True,
          since PEFT's attribute proxy makes direct .model() access unreliable.
        """
        is_peft = hasattr(self.temporal, "peft_type")

        if is_peft:
            # PEFT wrapping: access the underlying model directly via
            # base_model.model.model() to get only the last hidden state.
            # This avoids output_hidden_states=True which materializes ALL
            # intermediate layers (~394MB wasted for 16L 7B). Verified:
            # LoRA adapters remain active and results are identical (AF-101).
            base_out = self.temporal.base_model.model.model(input_ids=input_ids)
            temporal_hidden = base_out.last_hidden_state
            temporal_logits = self.temporal.base_model.model.lm_head(temporal_hidden)
        else:
            # Standard model — efficient path: only last hidden state materialized.
            base_out = self.temporal.model(input_ids=input_ids)
            temporal_hidden = base_out.last_hidden_state
            temporal_logits = self.temporal.lm_head(temporal_hidden)

        return temporal_hidden, temporal_logits

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_codebook_targets: torch.Tensor | None = None,
        frame_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            input_ids: Interleaved token sequence (x[:, :-1]). Shape: (batch, seq_len)
            audio_codebook_targets: Ground truth codebook tokens per audio frame.
                Shape: (batch, num_frames, num_codebooks)
                If None, only temporal logits are returned.
            frame_positions: Position of q0 for each frame in the FULL sequence.
                Shape: (num_frames,). Used to extract hidden states for the Depth
                Transformer. The hidden state at position (frame_pos - 1) in input_ids
                provides context for predicting that frame's codebooks.

        Returns:
            dict with:
                - 'temporal_logits': (batch, seq_len, vocab_size)
                - 'depth_logits': (batch, num_frames, num_codebooks, codebook_size)
                - 'temporal_hidden': last hidden states from Temporal Transformer
        """
        temporal_hidden, temporal_logits = self._temporal_forward(input_ids)

        result = {
            "temporal_logits": temporal_logits,
            "temporal_hidden": temporal_hidden,
        }

        # If we have audio targets, run Depth Transformer on audio frame positions
        if audio_codebook_targets is not None:
            seq_len = input_ids.size(1)

            # frame_positions are in FULL sequence coordinates.
            # input_ids = full_seq[:-1], so temporal_hidden[p] has processed full_seq[p].
            # To predict tokens at frame_positions[f], we use the hidden state just
            # BEFORE that position: hidden_positions = frame_positions - 1.
            hidden_positions = (frame_positions - 1).long()

            # Filter to valid positions within input_ids range
            valid_mask = (hidden_positions >= 0) & (hidden_positions < seq_len)
            hidden_positions = hidden_positions[valid_mask]
            num_valid = hidden_positions.size(0)
            audio_codebook_targets = audio_codebook_targets[:, :num_valid, :]

            depth_input = temporal_hidden[:, hidden_positions, :]  # (batch, num_frames, dim)

            # De-offset codebook targets from global token IDs to [0, codebook_size)
            # Global audio token = text_vocab_size + q * codebook_size + local_token
            cb_targets_local = audio_codebook_targets.clone()
            for q in range(self.config.num_codebooks):
                cb_targets_local[:, :, q] = (
                    cb_targets_local[:, :, q]
                    - self.config.text_vocab_size
                    - q * self.config.codebook_size
                )
            cb_targets_local = cb_targets_local.clamp(0, self.config.codebook_size - 1)

            # Run Depth Transformer with teacher forcing
            depth_logits = self.depth(depth_input, cb_targets_local)
            result["depth_logits"] = depth_logits
            result["cb_targets_local"] = cb_targets_local

        return result

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Sample a token from logits with temperature and top-k filtering."""
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            top_k_vals, _ = logits.topk(top_k, dim=-1)
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate one timestep: text token + codebook tokens for the model stream.

        For streaming inference. Takes the current context and generates
        the next window of tokens. In dual-stream mode, the input_ids should
        already include the user stream tokens for this window.

        Args:
            input_ids: Current context. Shape: (batch, context_len)
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            dict with:
                - 'text_token': (batch, 1) — predicted model text token
                - 'audio_tokens': (batch, num_codebooks) — predicted model codebook tokens
        """
        temporal_hidden, temporal_logits = self._temporal_forward(input_ids)

        # Sample model text token from last position's logits
        text_logits = temporal_logits[:, -1, :self.config.text_vocab_size]
        text_token = self._sample_token(text_logits, temperature, top_k)

        # Use last hidden state for depth transformer to predict codebook tokens
        h_last = temporal_hidden[:, -1, :]  # (batch, dim)
        audio_tokens = self.depth.generate(h_last, temperature=temperature, top_k=top_k)

        return {
            "text_token": text_token,
            "audio_tokens": audio_tokens,
        }

    def set_language(self, language_code: str) -> str:
        """Activate the language-family adapter for the given language.

        Args:
            language_code: ISO 639 language code (e.g., "kn", "hi", "en")

        Returns:
            The activated family name (or "english" for base model)
        """
        if not self.config.language_adapters:
            return "english"
        return activate_adapter(self.temporal, language_code)

    def param_count(self) -> dict[str, int]:
        """Return parameter counts for each component."""
        temporal_params = sum(p.numel() for p in self.temporal.parameters())
        depth_params = sum(p.numel() for p in self.depth.parameters())
        result = {
            "temporal": temporal_params,
            "depth": depth_params,
            "total": temporal_params + depth_params,
        }
        if self.config.language_adapters:
            result["adapters"] = get_adapter_info(self.temporal)
        return result
