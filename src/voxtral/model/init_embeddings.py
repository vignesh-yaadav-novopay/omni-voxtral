"""Embedding initialization for OmniVoxtral.

When extending Mistral's vocabulary from 32K to 81,920 tokens, we need careful
initialization to preserve pretrained knowledge:

1. Tokens [0, 32767]: Copy from Mistral's pretrained embeddings (English text)
2. Tokens [32768, 65535]: New Indic text tokens — init from Mistral embedding statistics
3. Tokens [65536, 81919]: Audio codebook tokens — init from Mimi's existing embeddings
   (or from pretrained embedding statistics if Mimi embeddings aren't available)

Reference: Moshi paper §4, Training Strategy doc Phase 1.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def initialize_extended_embeddings(
    model: nn.Module,
    original_vocab_size: int = 32768,
    text_vocab_size: int = 65536,
    total_vocab_size: int = 81920,
    num_codebooks: int = 8,
    codebook_size: int = 2048,
) -> dict[str, float]:
    """Initialize extended embedding and LM head weights.

    After `model.resize_token_embeddings(total_vocab_size)`, the new token embeddings
    are randomly initialized. This function replaces those random values with
    principled initialization that preserves training stability.

    Args:
        model: MistralForCausalLM (or similar) with already-resized embeddings
        original_vocab_size: Mistral's original vocab size (32768)
        text_vocab_size: Extended text vocab including Indic tokens (65536)
        total_vocab_size: Full vocab including audio tokens (81920)
        num_codebooks: Number of Mimi codebooks (8)
        codebook_size: Mimi codebook vocabulary (2048)

    Returns:
        dict with initialization statistics
    """
    embed_tokens = model.model.embed_tokens
    lm_head = model.lm_head

    # Get statistics from pretrained embeddings
    with torch.no_grad():
        pretrained_weights = embed_tokens.weight[:original_vocab_size]
        pretrained_mean = pretrained_weights.mean().item()
        pretrained_std = pretrained_weights.std().item()

        logger.info(
            f"Pretrained embedding stats: mean={pretrained_mean:.6f}, std={pretrained_std:.6f}"
        )

        # --- Initialize new text tokens (Indic) ---
        # Use same distribution as pretrained, but slightly smaller std
        # to avoid overwhelming the pretrained representation space
        indic_std = pretrained_std * 0.5
        num_new_text = text_vocab_size - original_vocab_size
        embed_tokens.weight[original_vocab_size:text_vocab_size].normal_(
            0, indic_std
        )
        logger.info(
            f"Initialized {num_new_text} Indic text tokens with std={indic_std:.6f}"
        )

        # --- Initialize audio codebook tokens ---
        # Audio tokens need to be distinguishable from text tokens
        # Use slightly larger std to create separation in embedding space
        audio_std = pretrained_std * 0.25
        num_audio = total_vocab_size - text_vocab_size
        embed_tokens.weight[text_vocab_size:total_vocab_size].normal_(
            0, audio_std
        )

        # Add per-codebook bias so different codebooks occupy different regions
        for q in range(num_codebooks):
            start_idx = text_vocab_size + q * codebook_size
            end_idx = start_idx + codebook_size
            # Small offset in a random direction for each codebook
            codebook_offset = torch.randn(embed_tokens.weight.size(1)) * 0.01
            embed_tokens.weight[start_idx:end_idx] += codebook_offset.to(
                embed_tokens.weight.device
            )

        logger.info(
            f"Initialized {num_audio} audio tokens ({num_codebooks} codebooks × "
            f"{codebook_size}) with std={audio_std:.6f}"
        )

        # --- Initialize LM head for new tokens ---
        # The LM head projects hidden states to vocab logits
        # For new tokens, initialize with small weights so initial logits are near-zero
        # (model won't predict new tokens until trained)
        if lm_head.weight.size(0) == total_vocab_size:
            lm_head_std = 0.01
            lm_head.weight[original_vocab_size:].normal_(0, lm_head_std)
            logger.info(
                f"Initialized LM head for new tokens with std={lm_head_std}"
            )

    stats = {
        "pretrained_mean": pretrained_mean,
        "pretrained_std": pretrained_std,
        "indic_std": indic_std,
        "audio_std": audio_std,
        "num_new_text": num_new_text,
        "num_audio": num_audio,
    }
    return stats
