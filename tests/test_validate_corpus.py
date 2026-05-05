"""Phase 1.5 — verify validate_v2_corpus.py catches the regressions it should.

This is the test that would have caught both bugs we hit during Phase 1a launch:
- Missing SP language control token at .npy[0]
- valid_token_mask length not matching token_count
"""

import json

import numpy as np
import pytest

from voxtral.data.sidecar import (
    atomic_save_npy,
    write_metadata_sidecar,
)


def _write_pair(tmp_path, lang, first_token, mask_len_offset=0):
    """Create a (.npy, .meta.json) pair under tmp_path/fleurs/<lang>/<shard>/.

    `first_token` is what we put as tokens[0] (pretending to be the SP control
    token).  `mask_len_offset` lets the test inject a length mismatch.
    """
    shard = lang[:2]
    d = tmp_path / "fleurs" / lang / shard
    d.mkdir(parents=True, exist_ok=True)
    tokens = np.concatenate([np.array([first_token]), np.arange(50, dtype=np.int64)])
    npy = d / f"{lang}_00000001_abc.npy"
    atomic_save_npy(tokens.astype(np.int64), str(npy))
    mask = [True] * (len(tokens) + mask_len_offset)
    write_metadata_sidecar(
        str(npy),
        {"language": lang, "source": "fleurs", "stream_layout": "single",
         "tokenizer_config": {"stride": 21}, "token_count": len(tokens)},
        valid_token_mask=mask,
    )
    return npy


def test_validate_passes_when_lang_token_correct(tmp_path):
    """Happy path: lang token at [0], mask length matches → 100% / 100%."""
    from scripts.validate_v2_corpus import validate_corpus, _load_sp_lang_tokens
    from pathlib import Path

    sp_path = Path("data/tokenizer/omnivoxtral_sp.model")
    if not sp_path.exists():
        pytest.skip("SP model not found")
    sp_tokens = _load_sp_lang_tokens(sp_path)
    hin_id = sp_tokens["hin"]

    _write_pair(tmp_path, "hin", first_token=hin_id)
    _write_pair(tmp_path, "tam", first_token=sp_tokens["tam"])

    rep = validate_corpus(tmp_path, sp_tokens)
    assert rep["hin"]["lang_token_ok"] == 1
    assert rep["hin"]["mask_ok"] == 1
    assert rep["tam"]["lang_token_ok"] == 1


def test_validate_flags_missing_lang_token(tmp_path):
    """Regression test: we caught the bug where the tokenizer wasn't injecting
    the SP control token. If the .npy starts with anything other than the
    expected lang_token id, validator must NOT count it as ok."""
    from scripts.validate_v2_corpus import validate_corpus, _load_sp_lang_tokens
    from pathlib import Path

    sp_path = Path("data/tokenizer/omnivoxtral_sp.model")
    if not sp_path.exists():
        pytest.skip("SP model not found")
    sp_tokens = _load_sp_lang_tokens(sp_path)

    _write_pair(tmp_path, "hin", first_token=42)  # NOT the lang token id

    rep = validate_corpus(tmp_path, sp_tokens)
    assert rep["hin"]["total"] == 1
    assert rep["hin"]["lang_token_ok"] == 0
    assert any("first token" in err for err in rep["hin"]["errors"])


def test_validate_flags_mask_length_mismatch(tmp_path):
    """Sidecar's valid_token_mask len must equal token_count. If they drift,
    the trainer's masking would silently drop the wrong positions."""
    from scripts.validate_v2_corpus import validate_corpus, _load_sp_lang_tokens
    from pathlib import Path

    sp_path = Path("data/tokenizer/omnivoxtral_sp.model")
    if not sp_path.exists():
        pytest.skip("SP model not found")
    sp_tokens = _load_sp_lang_tokens(sp_path)
    hin_id = sp_tokens["hin"]

    _write_pair(tmp_path, "hin", first_token=hin_id, mask_len_offset=-5)

    rep = validate_corpus(tmp_path, sp_tokens)
    assert rep["hin"]["mask_ok"] == 0
    assert any("mask len" in err for err in rep["hin"]["errors"])


def test_validate_flags_orphan_npy(tmp_path):
    """A .npy with no sidecar must show up under __no_sidecar__."""
    from scripts.validate_v2_corpus import validate_corpus

    d = tmp_path / "fleurs" / "hin" / "hi"
    d.mkdir(parents=True)
    atomic_save_npy(np.arange(10, dtype=np.int64), str(d / "orphan.npy"))

    rep = validate_corpus(tmp_path, {})
    assert rep["__no_sidecar__"]["total"] == 1
