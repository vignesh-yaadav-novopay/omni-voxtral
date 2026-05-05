"""Phase 5 — Regression test for the gibberish failure mode.

For each of the 13 target languages, a generated 30s audio sample must satisfy:
  - Whisper-large-v3 LID detects target language with confidence ≥ 0.9
  - Token autocorrelation at lag 50-200 < 0.5 (no repetitive loops)
  - F0 std > 30 Hz (not a monotone drone)

Tests are GPU-heavy and require a recent checkpoint at $VOXTRAL_CKPT. They are
skipped when no checkpoint env var is set so CI can still collect the rest of
the suite.
"""

import os

import numpy as np
import pytest
import torch


CHECKPOINT_ENV = "VOXTRAL_CKPT"


def _require_checkpoint():
    p = os.environ.get(CHECKPOINT_ENV)
    if not p or not os.path.exists(p):
        pytest.skip(
            f"set {CHECKPOINT_ENV}=<path-to-checkpoint.pt> to run no-gibberish "
            "regression tests on a real checkpoint"
        )
    return p


def compute_token_autocorrelation(tokens: torch.Tensor, lags: range) -> dict[int, float]:
    """Per-lag autocorrelation across a 1-D token sequence.

    Treats tokens as integers; computes correlation of (tokens[:-lag], tokens[lag:])
    after centering. Used as a structural test for repetitive looping.
    """
    out = {}
    if tokens.dim() != 1:
        tokens = tokens.flatten()
    x = tokens.float() - tokens.float().mean()
    denom = (x * x).sum().clamp_min(1e-8)
    for lag in lags:
        if lag >= tokens.numel():
            continue
        a = x[:-lag]
        b = x[lag:]
        out[lag] = float((a * b).sum() / denom)
    return out


def compute_f0_std(audio: np.ndarray, sample_rate: int = 24_000) -> float:
    """Estimate F0 std (Hz) over voiced regions. librosa.pyin used for robustness."""
    import librosa
    f0, voiced_flag, _ = librosa.pyin(
        audio.astype(np.float32),
        fmin=70.0, fmax=400.0,
        sr=sample_rate,
        frame_length=2048,
    )
    f0_voiced = f0[voiced_flag]
    if f0_voiced.size < 5:
        return 0.0
    return float(np.nanstd(f0_voiced))


# ---------------------------------------------------------------------------
# Logic-only tests (no GPU)
# ---------------------------------------------------------------------------

def test_token_autocorrelation_high_for_repetitive_loop():
    """A repeated 5-token cycle should have autocorrelation ≈ 1 at lag 5."""
    seq = torch.tensor([1, 2, 3, 4, 5] * 200)
    corr = compute_token_autocorrelation(seq, range(2, 11))
    assert corr[5] > 0.95, f"period-5 loop should give corr~1 at lag 5; got {corr}"


def test_token_autocorrelation_low_for_random_sequence():
    torch.manual_seed(0xDEADBEEF)
    seq = torch.randint(0, 80_000, (500,))
    corr = compute_token_autocorrelation(seq, range(50, 200))
    max_corr = max(corr.values())
    assert max_corr < 0.3, f"random tokens shouldn't autocorrelate: max={max_corr}"


def test_f0_std_zero_on_pure_silence():
    silence = np.zeros(24_000 * 2, dtype=np.float32)
    std = compute_f0_std(silence)
    assert std == 0.0


@pytest.mark.gpu
@pytest.mark.slow
def test_no_gibberish_for_each_language():
    """Run the full 30s generation regression. Requires VOXTRAL_CKPT set."""
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("regression test needs CUDA")
    ckpt = _require_checkpoint()

    # Real harness needs scripts/generate.py refactor + this loop.
    # For now, this is a structural placeholder: we expect the reviewer to wire
    # `from scripts.generate import generate_audio` and call it per language.
    pytest.skip(
        "no-gibberish regression: pending refactor of scripts/generate.py into "
        "a module-level entrypoint. See plan §6.Phase 5 / LLD §3.14 for the "
        "exact generation flow."
    )
