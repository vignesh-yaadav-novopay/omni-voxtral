"""Phase 1 tokenizer tests.

Tests that don't require GPU/Whisper-large weights run inline. Tests that need
the real Whisper+Mimi stack are marked `@pytest.mark.gpu` and skip when CUDA is
absent — the Phase 1a re-tokenization run exercises them on the real machine.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from voxtral.data.sidecar import (
    SCHEMA_VERSION,
    atomic_save_npy,
    atomic_write_json,
    derive_valid_token_mask_from_audio_length,
    new_run_id,
    read_metadata_sidecar,
    write_metadata_sidecar,
)
from voxtral.tokenizer.model import (
    VoxtralTokenizerConfig,
    _DEFAULT_SP_MODEL,
    iso3_to_whisper_code,
    normalize_language_to_iso3,
)


# ---------------------------------------------------------------------------
# ISO 639-1 ↔ ISO 639-3 — added in Phase 1 fix
# ---------------------------------------------------------------------------

def test_normalize_language_iso1_to_iso3():
    assert normalize_language_to_iso3("hi") == "hin"
    assert normalize_language_to_iso3("ta") == "tam"
    assert normalize_language_to_iso3("bn") == "ben"
    assert normalize_language_to_iso3("en") == "eng"


def test_normalize_language_idempotent_for_iso3():
    """Already-639-3 inputs should stay unchanged — caller didn't have to know."""
    assert normalize_language_to_iso3("hin") == "hin"
    assert normalize_language_to_iso3("brx") == "brx"
    assert normalize_language_to_iso3("eng") == "eng"


def test_normalize_language_rejects_unknown():
    """An unknown code must raise rather than silently mapping to <unk>."""
    import pytest as _pytest
    with _pytest.raises(ValueError):
        normalize_language_to_iso3("xx")
    with _pytest.raises(ValueError):
        normalize_language_to_iso3("unknown")


def test_iso3_to_whisper_returns_none_for_mms_only_langs():
    """The 7 Indic languages routed via MMS — Whisper has no support."""
    for mms in ["brx", "doi", "kok", "mni", "sat", "snd", "kas"]:
        assert iso3_to_whisper_code(mms) is None, f"{mms} should not map to Whisper"


def test_iso3_to_whisper_returns_iso1_for_supported():
    assert iso3_to_whisper_code("hin") == "hi"
    assert iso3_to_whisper_code("tam") == "ta"
    assert iso3_to_whisper_code("ben") == "bn"
    assert iso3_to_whisper_code("eng") == "en"


# ---------------------------------------------------------------------------
# Config defaults — Phase 1 spec
# ---------------------------------------------------------------------------

def test_default_sp_tokenizer_path_resolves_to_v2_model():
    cfg = VoxtralTokenizerConfig()
    assert cfg.sp_tokenizer_path == _DEFAULT_SP_MODEL
    assert cfg.sp_tokenizer_path == "data/tokenizer/omnivoxtral_sp.model"


def test_default_sp_model_file_exists_on_disk():
    """The SP model must actually be present at the default path."""
    sp_path = Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "omnivoxtral_sp.model"
    assert sp_path.exists(), (
        f"SP model missing at {sp_path}; Phase 1 retokenization will crash. "
        "Run scripts/train_tokenizer.py if you intend to regenerate it."
    )


def test_default_language_is_none_not_en():
    """Defect 1 fix: `language` no longer silently defaults to 'en'."""
    cfg = VoxtralTokenizerConfig()
    assert cfg.language is None


def test_explicit_none_sp_path_still_allowed_for_legacy_runs():
    """Mistral BPE remains opt-in via explicit None."""
    cfg = VoxtralTokenizerConfig(sp_tokenizer_path=None)
    assert cfg.sp_tokenizer_path is None


# ---------------------------------------------------------------------------
# Sidecar I/O — atomic writes, schema versioning
# ---------------------------------------------------------------------------

def test_sidecar_writes_schema_v2(tmp_path):
    npy = tmp_path / "ab" / "sample.npy"
    npy.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.zeros(10, dtype=np.int64), str(npy))
    write_metadata_sidecar(str(npy), {"language": "hi", "transcript": "नमस्ते"})
    side = npy.parent / "sample.meta.json"
    assert side.exists()
    payload = json.loads(side.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["language"] == "hi"
    assert payload["transcript"] == "नमस्ते"
    assert "preprocessing_run_id" in payload
    assert "preprocessing_timestamp" in payload


def test_sidecar_writes_atomically_via_tmp(tmp_path, monkeypatch):
    """If the rename step fails, the .tmp must not be left visible as the final."""
    npy = tmp_path / "x" / "atomic.npy"
    npy.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.zeros(2, dtype=np.int64), str(npy))

    # Inject failure at os.replace by monkeypatching it for the json call.
    real_replace = os.replace
    state = {"json_calls": 0}

    def crashy_replace(src, dst):
        if dst.endswith(".meta.json"):
            state["json_calls"] += 1
            raise OSError("simulated failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", crashy_replace)
    with pytest.raises(OSError):
        write_metadata_sidecar(str(npy), {"language": "hi"})
    sidecar_path = npy.parent / "atomic.meta.json"
    assert not sidecar_path.exists(), "sidecar must not exist after failed replace"


def test_read_sidecar_rejects_wrong_schema_version(tmp_path):
    npy = tmp_path / "y" / "wrong_schema.npy"
    npy.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.zeros(2, dtype=np.int64), str(npy))
    sidecar = npy.parent / "wrong_schema.meta.json"
    atomic_write_json({"schema_version": 99, "language": "hi"}, str(sidecar))
    with pytest.raises(ValueError, match="schema_version"):
        read_metadata_sidecar(str(npy))


def test_read_sidecar_returns_none_when_missing(tmp_path):
    npy = tmp_path / "z" / "no_meta.npy"
    npy.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.zeros(2, dtype=np.int64), str(npy))
    assert read_metadata_sidecar(str(npy)) is None


def test_valid_token_mask_derives_correctly_for_short_audio():
    # 10s of audio in a 20s chunk → real audio is the first 50% of windows.
    mask = derive_valid_token_mask_from_audio_length(
        audio_samples=10 * 24_000,
        chunk_samples=20 * 24_000,
        token_count=100,  # 100 // 21 = 4 windows; 50% = 2 real → 42 tokens
        stride=21,
    )
    assert sum(mask) <= 100
    # Most of the latter half must be False (zero-pad mask).
    assert not all(mask)
    assert mask[0] is True


def test_valid_token_mask_all_true_when_audio_fills_chunk():
    mask = derive_valid_token_mask_from_audio_length(
        audio_samples=20 * 24_000,
        chunk_samples=20 * 24_000,
        token_count=420,
        stride=21,
    )
    assert all(mask)
    assert len(mask) == 420


def test_run_id_is_unique_per_call():
    a = new_run_id()
    b = new_run_id()
    assert a != b
    assert len(a) == 32  # uuid4 hex


# ---------------------------------------------------------------------------
# valid_token_mask in compute_omni_loss — Phase 1.5 contract
# ---------------------------------------------------------------------------

def test_compute_omni_loss_accepts_valid_token_mask_kwarg():
    """The trainer must accept valid_token_mask without exploding on legacy code paths."""
    from voxtral.trainer.omni_trainer import compute_omni_loss
    import inspect

    sig = inspect.signature(compute_omni_loss)
    assert "valid_token_mask" in sig.parameters
    assert sig.parameters["valid_token_mask"].default is None  # back-compat


# ---------------------------------------------------------------------------
# VoxtralDataset sidecar reader behavior
# ---------------------------------------------------------------------------

def _make_corpus(tmp_path, n: int = 20, with_sidecars: bool = True, language: str = "hi"):
    for i in range(n):
        d = tmp_path / f"{i:02d}"[-2:]
        d.mkdir(parents=True, exist_ok=True)
        npy = d / f"f{i}.npy"
        atomic_save_npy(np.arange(100, dtype=np.int64), str(npy))
        if with_sidecars:
            write_metadata_sidecar(str(npy), {
                "language": language, "duration_s": 12.4, "source": "fleurs",
                "stream_layout": "single", "valid_token_mask": [True] * 100,
                "tokenizer_config": {"stride": 21},
            })


def test_dataset_yields_language_field_when_sidecar_exists(tmp_path):
    from voxtral.trainer.config import VoxtralTrainConfig
    from voxtral.trainer.data import VoxtralDataset

    _make_corpus(tmp_path, n=20, with_sidecars=True, language="hi")
    cfg = VoxtralTrainConfig(fake=False, data_path=str(tmp_path), max_steps=1)
    ds = VoxtralDataset(cfg, split="train")
    item = next(iter(ds))
    assert item["language"] == "hi"
    assert item["source"] == "fleurs"
    assert item["duration_s"] == 12.4
    assert item["stream_layout"] == "single"
    assert item["valid_token_mask"].dtype == torch.bool
    assert item["valid_token_mask"].numel() == 100


def test_dataset_falls_back_to_unknown_when_sidecar_missing(tmp_path):
    from voxtral.trainer.config import VoxtralTrainConfig
    from voxtral.trainer.data import VoxtralDataset

    _make_corpus(tmp_path, n=20, with_sidecars=False)
    cfg = VoxtralTrainConfig(fake=False, data_path=str(tmp_path), max_steps=1,
                             # Loosen threshold so the iterator yields rather than tripping.
                             )
    # Use a high threshold so the 1k cadence doesn't trip in this test.
    ds = VoxtralDataset(cfg, split="train", sidecar_missing_threshold=0.99)
    item = next(iter(ds))
    assert item["language"] == "unknown"
    assert item["valid_token_mask"].all()


def test_dataset_fails_fast_at_5pct_missing_sidecars(tmp_path):
    """If >5% of consumed files lack sidecars, raise instead of silently corrupting."""
    from voxtral.trainer.config import VoxtralTrainConfig
    from voxtral.trainer.data import VoxtralDataset

    # 1100 files, all without sidecars (100% missing rate). Trip after the
    # _meta_check_every=1000 cadence.
    for i in range(1100):
        d = tmp_path / f"{i:02d}"[-2:]
        d.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(np.arange(10, dtype=np.int64), str(d / f"f{i}.npy"))

    cfg = VoxtralTrainConfig(fake=False, data_path=str(tmp_path), max_steps=1)
    ds = VoxtralDataset(cfg, split="train", sidecar_missing_threshold=0.05)
    it = iter(ds)
    with pytest.raises(RuntimeError, match="lack sidecars"):
        for _ in range(1500):
            next(it)
