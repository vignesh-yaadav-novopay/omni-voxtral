"""Phase 3 — verify the dual-LID agreement + quarantine logic.

Uses fake primary/secondary LID models (no GPU, no model download) to
exercise:
- ISO 639-1 → 639-3 normalization
- agreement=True when primary and secondary match
- agreement=False when they disagree → quarantine
- low confidence → quarantine even on agreement
"""

import json

import numpy as np
import pytest
import torch

from voxtral.data.sidecar import atomic_save_npy


class _FakeLid:
    """Stand-in for _MMSLid / faster-whisper. Returns whatever was set."""
    def __init__(self, lang: str, conf: float):
        self.lang = lang
        self.conf = conf

    def detect(self, audio: torch.Tensor, sr: int):
        return (self.lang, self.conf)

    def __call__(self, path: str):
        return (self.lang, self.conf)


def _write_chunk(tmp_path, duration_s=10.0):
    """Lay down a fake chunk with .wav (silent, 16 kHz) + .json sidecar."""
    import torchaudio
    sr = 16_000
    samples = int(duration_s * sr)
    wav_path = tmp_path / "chunk.wav"
    torchaudio.save(str(wav_path), torch.zeros(1, samples), sr)
    json_path = tmp_path / "chunk.json"
    json_path.write_text(json.dumps({
        "duration_s": duration_s, "source_id": "fake", "chunk_index": 0,
    }))
    return wav_path


def test_normalize_lang_iso1_to_iso3():
    from scripts.detect_language import _normalize_lang
    assert _normalize_lang("hi") == "hin"
    assert _normalize_lang("ta") == "tam"
    # Already 639-3 stays unchanged.
    assert _normalize_lang("hin") == "hin"


def test_detect_returns_agreement_when_both_lids_match(tmp_path):
    from scripts.detect_language import detect_chunk_language

    wav = _write_chunk(tmp_path, duration_s=10.0)
    primary = _FakeLid("hin", 0.9)   # already iso3
    secondary = _FakeLid("hi", 0.85)  # iso1 → normalized to hin
    out = detect_chunk_language(
        str(wav), primary, secondary,
        confidence_threshold=0.7,
        streaming_context_threshold_s=4.0,  # 10s clip → no streaming context
    )
    assert out["language"] == "hin"
    assert out["agreement"] is True
    assert out["method"] == "mms_lid_2048+whisper_lid"
    assert out["secondary"]["language"] == "hin"


def test_detect_flags_disagreement(tmp_path):
    from scripts.detect_language import detect_chunk_language

    wav = _write_chunk(tmp_path, duration_s=10.0)
    primary = _FakeLid("hin", 0.95)
    secondary = _FakeLid("ta", 0.92)  # different language
    out = detect_chunk_language(
        str(wav), primary, secondary,
        confidence_threshold=0.7,
        streaming_context_threshold_s=4.0,
    )
    assert out["agreement"] is False
    assert out["method"] == "tie_break"


def test_detect_method_when_no_secondary(tmp_path):
    from scripts.detect_language import detect_chunk_language

    wav = _write_chunk(tmp_path, duration_s=10.0)
    primary = _FakeLid("ben", 0.8)
    out = detect_chunk_language(
        str(wav), primary, None,
        confidence_threshold=0.7,
        streaming_context_threshold_s=4.0,
    )
    assert out["language"] == "ben"
    assert out["method"] == "mms_lid_2048"
    assert out["secondary"] is None
    assert out["agreement"] is None


def test_detect_records_low_confidence(tmp_path):
    """Low-conf result is still returned (the main loop is what quarantines).
    But the returned dict must carry the low confidence so the caller can decide.
    """
    from scripts.detect_language import detect_chunk_language

    wav = _write_chunk(tmp_path, duration_s=10.0)
    primary = _FakeLid("hin", 0.4)  # below 0.7 threshold
    out = detect_chunk_language(
        str(wav), primary, None,
        confidence_threshold=0.7,
        streaming_context_threshold_s=4.0,
    )
    assert out["confidence"] == 0.4
    assert out["language"] == "hin"
