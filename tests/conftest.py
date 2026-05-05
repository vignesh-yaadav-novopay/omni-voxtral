"""Synthetic-audio fixtures shared across the v2 test suite.

Heavy fixtures (Whisper, Mimi, SP tokenizer) are module-scoped — building Whisper
once per test would dominate the suite. Audio fixtures are simple sinusoids /
noise / silence at 24 kHz, deterministic by torch.manual_seed.

Real-Indic-language audio fixture (`fleurs_hindi_clip`) is opt-in via env
`VOXTRAL_FLEURS_FIXTURE_PATH=/path/to/hi_clip.wav`. Tests that need real Indic
audio gracefully `pytest.skip` if the env var isn't set so the suite still runs
on a fresh machine.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SP_MODEL_PATH = REPO_ROOT / "data" / "tokenizer" / "omnivoxtral_sp.model"


def _mono_24k(duration_s: float, *, freq: float = 220.0, amplitude: float = 0.3) -> torch.Tensor:
    sr = 24_000
    n = int(duration_s * sr)
    t = torch.linspace(0.0, duration_s, n, dtype=torch.float32)
    wave = amplitude * torch.sin(2 * math.pi * freq * t)
    # (1, 1, n)
    return wave.view(1, 1, -1)


@pytest.fixture(scope="session")
def session_seed():
    torch.manual_seed(0xC0FFEE)
    np.random.seed(0xC0FFEE)
    return 0xC0FFEE


@pytest.fixture
def silence_5s(session_seed) -> torch.Tensor:
    """5s of bit-zero silence at 24kHz mono. Phase 0 experiment 2 input."""
    return torch.zeros(1, 1, 5 * 24_000, dtype=torch.float32)


@pytest.fixture
def noise_5s_minus_45db(session_seed) -> torch.Tensor:
    """5s of -45 dBFS Gaussian white noise at 24kHz mono."""
    n = 5 * 24_000
    raw = torch.randn(n, dtype=torch.float32)
    rms = raw.pow(2).mean().sqrt()
    target_rms = 10 ** (-45 / 20)  # -45 dBFS
    scaled = raw * (target_rms / rms.clamp_min(1e-8))
    return scaled.view(1, 1, -1)


@pytest.fixture
def pink_noise_5s_minus_45db(session_seed) -> torch.Tensor:
    """5s of -45 dBFS pink noise (1/f spectrum) at 24kHz mono."""
    n = 5 * 24_000
    # Voss-McCartney pink noise approximation
    rows = 16
    arr = np.random.randn(rows, n // rows + 1)
    cum = np.cumsum(arr, axis=1)
    pink = cum.sum(axis=0)[:n].astype(np.float32)
    pink = torch.from_numpy(pink)
    pink = pink - pink.mean()
    rms = pink.pow(2).mean().sqrt()
    target_rms = 10 ** (-45 / 20)
    scaled = pink * (target_rms / rms.clamp_min(1e-8))
    return scaled.view(1, 1, -1)


@pytest.fixture
def room_tone_5s(session_seed) -> torch.Tensor:
    """Synthetic room tone: -45 dB pink noise + faint 60 Hz hum.

    Stand-in for the Phase 0 'real room-tone from FLEURS quiet region' fixture
    when no FLEURS sample is on disk. Tests can override by setting
    `VOXTRAL_ROOM_TONE_PATH=/path/to/room_tone.wav`.
    """
    override = os.environ.get("VOXTRAL_ROOM_TONE_PATH")
    if override and os.path.exists(override):
        import soundfile as sf
        data, sr = sf.read(override)
        assert sr == 24_000
        if data.ndim > 1:
            data = data.mean(axis=1)
        # take the first 5s
        data = data[: 5 * 24_000]
        return torch.from_numpy(data.astype(np.float32)).view(1, 1, -1)
    n = 5 * 24_000
    arr = np.random.randn(8, n // 8 + 1)
    cum = np.cumsum(arr, axis=1)
    pink = cum.sum(axis=0)[:n].astype(np.float32)
    pink = pink - pink.mean()
    rms = pink.pow(2).mean().sqrt()
    target_rms = 10 ** (-45 / 20)
    pink = pink * (target_rms / rms.clamp_min(1e-8))
    t = np.arange(n) / 24_000
    hum = (10 ** (-55 / 20)) * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    return torch.from_numpy(pink + hum).view(1, 1, -1)


@pytest.fixture
def synthetic_two_speaker_audio(session_seed) -> tuple[torch.Tensor, list[dict]]:
    """30s of two-speaker turn-taking with known boundaries.

    Returns (audio[(1, 1, 30*24000)], segments[{speaker, start, end}]).
    Audio is sinusoidal at different fundamentals to mimic two voices; not a
    real speaker model. Phase 4 tests use it for boundary-check only.
    """
    sr = 24_000
    duration = 30.0
    n = int(duration * sr)
    out = torch.zeros(n, dtype=torch.float32)
    segments = [
        {"speaker": "S0", "start": 0.0, "end": 8.0, "freq": 220.0},
        {"speaker": "S1", "start": 8.5, "end": 15.0, "freq": 330.0},
        {"speaker": "S0", "start": 15.5, "end": 21.0, "freq": 220.0},
        {"speaker": "S1", "start": 21.5, "end": 28.0, "freq": 330.0},
    ]
    t_full = torch.linspace(0.0, duration, n, dtype=torch.float32)
    for seg in segments:
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        out[s:e] = 0.3 * torch.sin(2 * math.pi * seg["freq"] * t_full[s:e])
    seg_meta = [
        {"speaker": s["speaker"], "start": s["start"], "end": s["end"]}
        for s in segments
    ]
    return out.view(1, 1, -1), seg_meta


@pytest.fixture
def fleurs_hindi_clip() -> torch.Tensor:
    """5s real Hindi audio if available; otherwise pytest.skip.

    Set VOXTRAL_FLEURS_FIXTURE_PATH=/abs/path/to/hi_clip.wav (24kHz mono).
    """
    path = os.environ.get("VOXTRAL_FLEURS_FIXTURE_PATH")
    if not path or not os.path.exists(path):
        pytest.skip(
            "VOXTRAL_FLEURS_FIXTURE_PATH not set; skipping tests that need real "
            "Indic audio. Set the env var to a 24kHz Hindi WAV to enable."
        )
    import soundfile as sf
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 24_000:
        import torchaudio.functional as F
        data_t = torch.from_numpy(data.astype(np.float32)).view(1, -1)
        data_t = F.resample(data_t, sr, 24_000)
        data = data_t.numpy().squeeze()
    return torch.from_numpy(data.astype(np.float32)).view(1, 1, -1)


@pytest.fixture(scope="session")
def sp_tokenizer_path() -> str:
    if not SP_MODEL_PATH.exists():
        pytest.skip(f"SentencePiece model missing at {SP_MODEL_PATH}")
    return str(SP_MODEL_PATH)


# Lazy-import math to avoid loading at collection time.
import math  # noqa: E402
