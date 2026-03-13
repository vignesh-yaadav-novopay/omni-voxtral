"""OmniVoxtral Training Loop.

Extends the existing Voxtral trainer to support the dual-transformer architecture
(Temporal + Depth). The key differences from trainer.py:

1. Two loss components: temporal_loss (text + first codebook) + depth_loss (all codebooks)
2. Uses OmniVoxtral model instead of raw MistralForCausalLM
3. Extracts codebook targets from the interleaved token sequence for Depth Transformer
4. Separate optimizer groups for temporal (pretrained) and depth (from scratch)
"""

import copy
import dataclasses

import torch
import torch.nn.functional as F
import voxtral.trainer.utils as utils
from torch.nn.parallel import DistributedDataParallel
from voxtral.model.omnivoxtral import OmniVoxtral, OmniVoxtralConfig
from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import VoxtralDataset


@dataclasses.dataclass
class OmniTrainState:
    step: int
    model: OmniVoxtral | DistributedDataParallel
    ema: OmniVoxtral
    optimizer: torch.optim.AdamW  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler  # type: ignore
    train_dataset: VoxtralDataset


def extract_codebook_targets(
    tokens: torch.Tensor,
    stride: int,
    num_codebooks: int,
    dual_stream: bool = False,
    stream_stride: int = 21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-frame codebook targets from interleaved sequence.

    Each window has text_to_audio_factor audio tokens (e.g. 20 for stride=21).
    Since 20/8 = 2.5, frames can span window boundaries. This function correctly
    extracts ALL audio tokens, concatenates them, and reshapes into per-frame
    groups of num_codebooks tokens.

    Audio interleaving order: [q0_f1, q1_f1, ..., q7_f1, q0_f2, q1_f2, ...]
    So each group of num_codebooks consecutive audio tokens is one complete frame.

    Args:
        tokens: (batch, seq_len) — FULL interleaved token sequence (not target_ids)
        stride: tokens per window (21 single, 42 dual)
        num_codebooks: number of codebooks (8)
        dual_stream: whether the sequence uses dual-stream layout
        stream_stride: tokens per stream (21), only used when dual_stream=True

    Returns:
        codebook_targets: (batch, num_frames, num_codebooks) — correctly aligned
        frame_positions: (num_frames,) — position of q0 for each frame in the
            full interleaved sequence (for hidden state extraction)
    """
    batch_size, seq_len = tokens.shape
    num_windows = seq_len // stride

    # Reshape into windows: (batch, num_windows, stride)
    windowed = tokens[:, : num_windows * stride].view(batch_size, num_windows, stride)

    if dual_stream:
        # Model audio: positions stream_stride+1 through stride-1
        audio_slice = windowed[:, :, stream_stride + 1 : stride]
        model_offset = stream_stride + 1
    else:
        # Audio: positions 1 through stride-1
        audio_slice = windowed[:, :, 1 : stride]
        model_offset = 1

    # Flatten all audio tokens: (batch, num_windows * audio_per_window)
    audio_per_window = audio_slice.shape[2]
    flat_audio = audio_slice.reshape(batch_size, -1)

    # Reshape into frames: each group of num_codebooks is one complete frame
    num_audio_total = flat_audio.shape[1]
    num_frames = num_audio_total // num_codebooks
    codebook_targets = flat_audio[:, : num_frames * num_codebooks].view(
        batch_size, num_frames, num_codebooks
    )

    # Compute frame positions in the FULL interleaved sequence
    # Frame f's q0 is at flat_audio index f * num_codebooks
    # Map back to interleaved position: window_idx * stride + model_offset + offset
    frame_indices = torch.arange(num_frames, device=tokens.device)
    flat_pos = frame_indices * num_codebooks
    window_idx = flat_pos // audio_per_window
    offset_in_window = flat_pos % audio_per_window
    frame_positions = window_idx * stride + model_offset + offset_in_window

    return codebook_targets, frame_positions


def _build_stream_weight_pattern(
    text_w: float,
    semantic_w: float,
    acoustic_w: float,
    num_q: int,
    text_to_audio_factor: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build loss weight pattern for a single stream (text + audio tokens).

    Returns: (stream_stride,) tensor with weights per token position.
    """
    stream_stride = 1 + text_to_audio_factor
    pattern = torch.zeros(stream_stride, device=device, dtype=dtype)
    pattern[0] = text_w
    for i in range(1, stream_stride):
        audio_idx = i - 1
        if audio_idx % num_q == 0:
            pattern[i] = semantic_w
        else:
            pattern[i] = acoustic_w
    return pattern


def compute_omni_loss(
    model: OmniVoxtral | DistributedDataParallel,
    x: torch.Tensor,
    config: VoxtralTrainConfig,
) -> dict[str, torch.Tensor]:
    """Compute combined temporal + depth loss.

    Handles both single-stream and dual-stream token layouts.
    Returns dict with 'temporal_loss', 'depth_loss', and 'total_loss'.
    """
    omni = utils.unwrap_model(model)

    input_ids = x[:, :-1].contiguous()
    target_ids = x[:, 1:].contiguous()

    # Extract codebook targets from FULL sequence (not target_ids)
    # This ensures correct frame alignment regardless of the shift
    stride = omni.config.stride
    num_q = omni.config.num_codebooks
    is_dual = omni.config.dual_stream
    codebook_targets, frame_positions = extract_codebook_targets(
        x, stride, num_q,
        dual_stream=is_dual,
        stream_stride=omni.config.stream_stride,
    )

    # Forward through dual transformer with frame positions for hidden state extraction
    outputs = model(
        input_ids=input_ids,
        audio_codebook_targets=codebook_targets,
        frame_positions=frame_positions,
    )

    # --- Temporal Loss (weighted) ---
    temporal_logits = outputs["temporal_logits"]
    text_w, semantic_w, acoustic_w = config.loss_weights
    text_hz = config.voxtral_tokenizer_config.text_hz
    text_to_audio_factor = int(12.5 * num_q / text_hz)

    # Build per-stream weight pattern
    stream_pattern = _build_stream_weight_pattern(
        text_w, semantic_w, acoustic_w, num_q, text_to_audio_factor,
        device=temporal_logits.device, dtype=temporal_logits.dtype,
    )

    if is_dual:
        full_pattern = torch.cat([stream_pattern, stream_pattern])
    else:
        full_pattern = stream_pattern

    # Build weights aligned with target_ids (shifted by 1 from full sequence).
    # full_pattern[0] = text weight, but target_ids[0] = x[1] (an audio token).
    # Fix: build weights for FULL sequence, then take [1:] to align with target_ids.
    full_seq_len = x.size(1)
    repeats = (full_seq_len + len(full_pattern) - 1) // len(full_pattern)
    full_weights = full_pattern.repeat(repeats)[:full_seq_len]
    weights = full_weights[1:]  # Align with target_ids
    weights = weights / weights.mean()

    temporal_per_token = F.cross_entropy(
        temporal_logits.view(-1, temporal_logits.size(-1)).float(),
        target_ids.view(-1),
        reduction="none",
    )
    batch_size = target_ids.size(0)
    temporal_loss = (temporal_per_token * weights.repeat(batch_size)).mean()

    # --- Depth Loss (cross-entropy over codebook predictions) ---
    depth_logits = outputs["depth_logits"]
    cb_targets_local = outputs["cb_targets_local"]
    codebook_size = omni.config.codebook_size

    depth_loss = F.cross_entropy(
        depth_logits.reshape(-1, codebook_size).float(),
        cb_targets_local.reshape(-1),
    )

    # Combined loss: Moshi uses equal weighting
    total_loss = temporal_loss + depth_loss

    # --- Accuracy metrics (no extra memory, computed from existing tensors) ---
    with torch.no_grad():
        # Temporal: per-token-type accuracy
        temporal_preds = temporal_logits.argmax(dim=-1)  # (batch, seq_len)
        correct = (temporal_preds == target_ids).float()

        # Build position masks aligned with target_ids
        pos_in_pattern = torch.arange(target_ids.size(1), device=target_ids.device) + 1
        pos_in_stride = pos_in_pattern % len(full_pattern)
        text_mask = pos_in_stride == 0  # text positions
        audio_mask = ~text_mask

        text_acc = correct[:, text_mask].mean() if text_mask.any() else torch.tensor(0.0)
        audio_acc = correct[:, audio_mask].mean() if audio_mask.any() else torch.tensor(0.0)

        # Depth: per-codebook accuracy
        depth_preds = depth_logits.argmax(dim=-1)  # (batch, num_frames, num_codebooks)
        depth_correct = (depth_preds == cb_targets_local).float()
        depth_acc = depth_correct.mean()
        # First codebook (semantic) vs rest (acoustic)
        q0_acc = depth_correct[:, :, 0].mean()
        q1_7_acc = depth_correct[:, :, 1:].mean()

    return {
        "temporal_loss": temporal_loss,
        "depth_loss": depth_loss,
        "total_loss": total_loss,
        "text_acc": text_acc,
        "audio_acc": audio_acc,
        "depth_acc": depth_acc,
        "depth_q0_acc": q0_acc,
        "depth_q1_7_acc": q1_7_acc,
    }


def omni_train_step(
    state: OmniTrainState,
    batch: dict,
    stats: dict[str, float],
    config: VoxtralTrainConfig,
) -> tuple[OmniTrainState, dict[str, float]]:
    """Single training step for OmniVoxtral."""
    device = utils.get_device()
    x = batch["tokens"].to(device)

    losses = compute_omni_loss(state.model, x=x, config=config)

    state.optimizer.zero_grad()
    losses["total_loss"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), config.grad_norm
    )
    state.optimizer.step()
    state.scheduler.step()

    # Accumulate stats
    stats["count"] += 1
    stats["temporal_loss"] += losses["temporal_loss"].item()
    stats["depth_loss"] += losses["depth_loss"].item()
    stats["total_loss"] += losses["total_loss"].item()
    stats["grad_norm"] += grad_norm.item()
    stats["text_acc"] += losses["text_acc"].item()
    stats["audio_acc"] += losses["audio_acc"].item()
    stats["depth_acc"] += losses["depth_acc"].item()
    stats["depth_q0_acc"] += losses["depth_q0_acc"].item()
    stats["depth_q1_7_acc"] += losses["depth_q1_7_acc"].item()

    return state, stats
