"""Phase 4 — diarize_v2 + DualStreamTokenizer.encode_with_metadata tests.

GPU-heavy paths (real pyannote) are skipped without CUDA. Logic tests cover:
- silence-region replacement strategy outputs non-zero audio
- DualStreamTokenizer.encode_with_metadata emits stride-42 + dual layout sidecar
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_silence_strategy_noise_floor_produces_non_zero_audio():
    """Plan §6.Phase 4 contract: silence regions must NOT be bit-zero."""
    from scripts.diarize_v2 import _generate_silence

    n = 24_000
    out = _generate_silence(n, 24_000, "noise_floor")
    assert out.shape[-1] == n
    assert (out.abs() > 0).any(), "noise_floor strategy must produce non-zero samples"
    # And it should be near -45 dB RMS.
    rms = float(out.pow(2).mean().sqrt())
    target = 10 ** (-45 / 20)
    assert 0.5 * target <= rms <= 2.0 * target, f"RMS={rms:.5f} not near {target:.5f}"


def test_silence_strategy_room_tone_falls_back_when_no_template():
    """If room_tone_template is None, fall back to noise_floor; never crash."""
    from scripts.diarize_v2 import _generate_silence

    out = _generate_silence(24_000, 24_000, "room_tone", room_tone_template=None)
    assert out.numel() == 24_000
    assert (out.abs() > 0).any()


def test_mask_other_speaker_keeps_target_intact():
    """Target speaker's audio is preserved 1:1; non-target replaced with silence."""
    from scripts.diarize_v2 import _mask_other_speaker

    sr = 24_000
    n = 5 * sr
    full = torch.linspace(-1.0, 1.0, n).view(1, -1)  # known signal
    segments = [
        {"speaker": "S0", "start": 0.0, "end": 1.0},
        {"speaker": "S1", "start": 1.0, "end": 2.5},
        {"speaker": "S0", "start": 2.5, "end": 5.0},
    ]
    masked = _mask_other_speaker(full, sr, "S0", segments)
    # In S0 regions, output should match full.
    assert torch.allclose(masked[0, : sr], full[0, : sr], atol=1e-6)
    assert torch.allclose(
        masked[0, int(2.5 * sr):], full[0, int(2.5 * sr):], atol=1e-6
    )
    # In the S1 gap, output should differ from full but still be non-zero.
    s1_slice = masked[0, sr:int(2.5 * sr)]
    assert (s1_slice.abs() > 0).any()
    assert not torch.allclose(s1_slice, full[0, sr:int(2.5 * sr)], atol=1e-3)


@pytest.mark.gpu
def test_dual_stream_encode_with_metadata_emits_stride_42(synthetic_two_speaker_audio):
    """encode_with_metadata returns tokens of stride 42 and stream_layout='dual'."""
    if not torch.cuda.is_available():
        pytest.skip("DualStreamTokenizer needs CUDA")
    from voxtral.tokenizer.dual_stream import DualStreamTokenizer
    from voxtral.tokenizer.model import VoxtralTokenizerConfig

    audio, segments = synthetic_two_speaker_audio
    cfg = VoxtralTokenizerConfig()
    tok = DualStreamTokenizer(cfg).to("cuda").to(torch.float16)
    audio = audio.to("cuda").to(torch.float16)
    # User stream + model stream from the same single-speaker fixture for shape test
    tokens, meta = tok.encode_with_metadata(
        user_audio=audio, model_audio=audio,
        sample_rate=24_000, segments_metadata=segments,
        language="hin",
    )
    assert tokens.dim() == 2
    assert meta["stream_layout"] == "dual"
    assert meta["tokenizer_config"]["stride"] == 42
    assert meta["language"] == "hin"
    assert tokens.size(-1) % 42 == 0


def test_dual_chunk_meta_atomicity(tmp_path):
    """Smoke: _atomic_write_json never leaves .tmp files even if interrupted."""
    from scripts.diarize_v2 import _atomic_write_json

    p = tmp_path / "out.json"
    _atomic_write_json({"hello": "world"}, str(p))
    assert p.exists()
    assert not (tmp_path / "out.json.tmp").exists()
    assert json.loads(p.read_text())["hello"] == "world"
