"""Phase 5 — Goal 4 evaluation harness.

Synthetic dual-stream input has known user-stream speech regions; the model is
expected to emit silence tokens during ≥ 70% of those regions.

A pure-logic helper `interruption_silence_emission_rate` is unit-tested here
without touching the model. The integration test that wires the model is
GPU-heavy and skipped without VOXTRAL_CKPT.
"""

import os

import numpy as np
import pytest
import torch


def interruption_silence_emission_rate(
    model_stream_tokens: torch.Tensor,
    user_speech_mask: torch.Tensor,
    silence_token_id: int,
) -> float:
    """Fraction of model-stream positions that emit `silence_token_id` while
    `user_speech_mask` is True.

    model_stream_tokens: (T,) int — text tokens of the model stream over time
    user_speech_mask:     (T,) bool — True at frames where user is speaking
    Returns the silence-emission rate over user-speech frames; 0.0 if none.
    """
    assert model_stream_tokens.dim() == 1 and user_speech_mask.dim() == 1
    assert model_stream_tokens.size(0) == user_speech_mask.size(0)
    if not user_speech_mask.any():
        return 0.0
    silent_during_user = (model_stream_tokens[user_speech_mask] == silence_token_id)
    return float(silent_during_user.float().mean().item())


def test_perfect_silence_emission_returns_one():
    SILENCE = 27  # SP control token id
    n = 1000
    user_mask = torch.zeros(n, dtype=torch.bool)
    user_mask[100:600] = True
    model = torch.full((n,), SILENCE, dtype=torch.long)
    rate = interruption_silence_emission_rate(model, user_mask, SILENCE)
    assert rate == 1.0


def test_rate_returns_zero_when_user_never_speaks():
    SILENCE = 27
    n = 100
    rate = interruption_silence_emission_rate(
        torch.zeros(n, dtype=torch.long),
        torch.zeros(n, dtype=torch.bool),
        SILENCE,
    )
    assert rate == 0.0


def test_partial_emission_rate():
    SILENCE = 27
    user_mask = torch.tensor([True] * 10, dtype=torch.bool)
    # 7 silence + 3 non-silence tokens during user speech → 70%
    model = torch.tensor([SILENCE] * 7 + [42, 43, 44], dtype=torch.long)
    rate = interruption_silence_emission_rate(model, user_mask, SILENCE)
    assert abs(rate - 0.70) < 1e-6


def test_rate_above_seventy_percent_meets_goal4():
    """Spot-check the >=70% Goal 4 acceptance criterion."""
    SILENCE = 27
    user = torch.tensor([True] * 100, dtype=torch.bool)
    model = torch.cat([
        torch.full((75,), SILENCE, dtype=torch.long),
        torch.arange(25, dtype=torch.long) + 100,
    ])
    rate = interruption_silence_emission_rate(model, user, SILENCE)
    assert rate >= 0.70


@pytest.mark.gpu
@pytest.mark.slow
def test_model_emits_silence_during_user_speech():
    """Full eval: needs a dual-stream VOXTRAL_CKPT.

    Constructs a synthetic user-side waveform with speech in [4s, 24s] (frames
    100..600 at 5 Hz) and silence elsewhere. Generates 30s of model output.
    Expects the model to emit `<|silence|>` (SP id 27) on ≥70% of model-stream
    text positions while the user is speaking — Goal 4 acceptance.
    """
    if not os.environ.get("VOXTRAL_CKPT"):
        pytest.skip("set VOXTRAL_CKPT to run the integration eval")
    if not torch.cuda.is_available():
        pytest.skip("integration test needs CUDA")
    if os.environ.get("DUAL_STREAM") != "true":
        pytest.skip("integration test needs DUAL_STREAM=true (dual-stream checkpoint)")

    from scripts.generate import generate_dual_stream, load_inference_pipeline
    from voxtral.trainer.config import VoxtralTrainConfig

    sr = 24_000
    n = 30 * sr  # 30s
    user = torch.zeros(n)
    # Speech burst from 4s to 24s — sinusoid + noise so RMS clearly clears
    # the speech_rms_threshold.
    t = torch.arange(4 * sr, 24 * sr).float() / sr
    user[4 * sr : 24 * sr] = 0.1 * torch.sin(2 * 3.1415 * 200 * t) + 0.02 * torch.randn(20 * sr)

    config = VoxtralTrainConfig(dual_stream=True)
    pipeline = load_inference_pipeline(os.environ["VOXTRAL_CKPT"], config=config, device="cuda:0")

    out = generate_dual_stream(
        pipeline,
        user_audio=user.numpy(),
        user_sample_rate=sr,
        language="hin",
        max_windows=150,
        temperature=0.8,
    )

    rate = interruption_silence_emission_rate(
        out["model_text_tokens"], out["user_speech_mask"], silence_token_id=27,
    )
    assert rate >= 0.70, (
        f"Goal 4 acceptance fail: silence emission rate {rate:.2%} < 70% during "
        f"user speech. (model_tokens={out['model_text_tokens'][:20].tolist()}, "
        f"user_mask sum={int(out['user_speech_mask'].sum())})"
    )
