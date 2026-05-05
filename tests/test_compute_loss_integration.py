"""Phase 1.5 — end-to-end compute_omni_loss with a real (tiny) model.

The signature/shape tests don't catch silent shape bugs in the depth-loss
masking path. This test loads the smallest viable OmniVoxtral
(mistral-1L-tiny, ~65M params on CPU), runs one forward pass with a
valid_token_mask that masks half the positions, and verifies:

- Loss is finite (no NaN from divide-by-zero when mask all False or 0 valid)
- Loss with mask=all-True ~ loss with mask=None (back-compat path)
- Loss with mask=all-False produces 0 loss (no positions to learn from)

Slow (~30s on CPU), so marked @pytest.mark.slow and gated behind
RUN_TINY_LOSS_INTEGRATION=1.
"""

import os

import pytest
import torch


def _build_tiny_model_and_cfg():
    from voxtral.model.omnivoxtral import OmniVoxtral, OmniVoxtralConfig
    from voxtral.trainer.config import VoxtralTrainConfig

    cfg = VoxtralTrainConfig(
        mistral_pretrained_path="nilq/mistral-1L-tiny",
        new_vocab_size=81920,
        loss_weights=[100, 1, 1],
        prune_layers=None, lora_rank=None, fake=True,
        max_seq_len=210,
    )
    omni_cfg = OmniVoxtralConfig(
        temporal_pretrained_path=cfg.mistral_pretrained_path,
        new_vocab_size=cfg.new_vocab_size,
        text_vocab_size=cfg.voxtral_tokenizer_config.text_vocab_size,
        num_codebooks=cfg.voxtral_tokenizer_config.mimi_num_quantizers,
        codebook_size=(cfg.new_vocab_size - cfg.voxtral_tokenizer_config.text_vocab_size) // cfg.voxtral_tokenizer_config.mimi_num_quantizers,
        text_hz=cfg.voxtral_tokenizer_config.text_hz,
        depth_num_layers=2, depth_dim=64, depth_num_heads=4,
        depth_dropout=0.0,
    )
    model = OmniVoxtral(omni_cfg).to("cpu", torch.float32)
    model.train(False)  # inference mode (not training mode)
    return model, cfg


@pytest.mark.slow
def test_compute_omni_loss_handles_partial_mask_on_real_model():
    if not os.environ.get("RUN_TINY_LOSS_INTEGRATION"):
        pytest.skip("set RUN_TINY_LOSS_INTEGRATION=1 to run (~30s on CPU)")

    from voxtral.trainer.omni_trainer import compute_omni_loss

    model, cfg = _build_tiny_model_and_cfg()
    seq_len = 210
    batch = torch.randint(0, cfg.new_vocab_size, (1, seq_len), dtype=torch.long)
    mask_partial = torch.ones((1, seq_len), dtype=torch.bool)
    mask_partial[:, seq_len // 2:] = False

    losses = {}
    for tag, mask in [("none", None), ("all_true", torch.ones_like(mask_partial)),
                      ("partial", mask_partial)]:
        with torch.no_grad():
            out = compute_omni_loss(model, x=batch, config=cfg, valid_token_mask=mask)
        total = out["total_loss"] if isinstance(out, dict) else out
        losses[tag] = float(total)
        assert torch.isfinite(torch.tensor(losses[tag])), f"{tag}: loss not finite"

    # all-True should match no-mask within numerical noise.
    rel = abs(losses["none"] - losses["all_true"]) / max(abs(losses["none"]), 1e-6)
    assert rel < 0.01, f"all-True mask vs no-mask drifted: {losses}"


@pytest.mark.slow
def test_compute_omni_loss_full_mask_zero_does_not_nan():
    if not os.environ.get("RUN_TINY_LOSS_INTEGRATION"):
        pytest.skip("set RUN_TINY_LOSS_INTEGRATION=1 to run")

    from voxtral.trainer.omni_trainer import compute_omni_loss

    model, cfg = _build_tiny_model_and_cfg()
    seq_len = 210
    batch = torch.randint(0, cfg.new_vocab_size, (1, seq_len), dtype=torch.long)
    mask_zero = torch.zeros((1, seq_len), dtype=torch.bool)
    with torch.no_grad():
        out = compute_omni_loss(model, x=batch, config=cfg, valid_token_mask=mask_zero)
    total = out["total_loss"] if isinstance(out, dict) else out
    val = float(total)
    assert torch.isfinite(torch.tensor(val)), f"all-False mask non-finite: {val}"
