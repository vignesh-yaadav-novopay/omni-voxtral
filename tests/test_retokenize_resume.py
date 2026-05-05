"""Phase 1 — verify retokenize_v2.py's resume-safety helpers.

These run cold (no GPU, no Mimi load) — they only exercise the file-system
short-circuit logic that decides whether to skip already-tokenized files when
the multi-day retokenization job is restarted.
"""

import json

import numpy as np
import pytest

from voxtral.data.sidecar import (
    SCHEMA_VERSION,
    atomic_save_npy,
    write_metadata_sidecar,
)


def _make_done_chunk(out_root, source, lang, filename):
    """Helper: lay down both a .npy and a v2 sidecar at the path retokenize_v2
    would have written. Mimics a completed file from a prior run."""
    from scripts.retokenize_v2 import _output_paths

    _, npy_path, _meta_path = _output_paths(str(out_root), source, lang, filename)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.arange(50, dtype=np.int64), str(npy_path))
    write_metadata_sidecar(
        str(npy_path),
        {"language": lang, "source": source, "tokenizer_config": {"stride": 21}},
    )
    return npy_path


def test_already_tokenized_returns_true_when_both_files_exist(tmp_path):
    from scripts.retokenize_v2 import _already_tokenized

    _make_done_chunk(tmp_path, "fleurs", "hin", "hin_00000001_abc12345")
    assert _already_tokenized(str(tmp_path), "fleurs", "hin", "hin_00000001_abc12345")


def test_already_tokenized_returns_false_when_npy_missing(tmp_path):
    from scripts.retokenize_v2 import _already_tokenized

    # Sidecar without npy — partial state from a crash.
    from scripts.retokenize_v2 import _output_paths
    _, _npy, meta = _output_paths(str(tmp_path), "iv", "tam", "tam_99")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "language": "tam"}))
    assert not _already_tokenized(str(tmp_path), "iv", "tam", "tam_99")


def test_already_tokenized_returns_false_for_v1_schema(tmp_path):
    """A v1-schema sidecar (e.g. left over from old preprocessing) must NOT
    count as 'done' — we want it re-tokenized to v2."""
    from scripts.retokenize_v2 import _already_tokenized, _output_paths

    _, npy_path, meta_path = _output_paths(str(tmp_path), "fleurs", "ben", "ben_42")
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(np.arange(10, dtype=np.int64), str(npy_path))
    meta_path.write_text(json.dumps({"schema_version": 1, "language": "ben"}))
    assert not _already_tokenized(str(tmp_path), "fleurs", "ben", "ben_42")


def test_index_already_tokenized_globs_by_idx_prefix(tmp_path):
    """When the audio hash isn't known yet (we want to skip BEFORE loading the
    row), the cheaper index-based check globs on `<lang>_<idx:08d>_*`."""
    from scripts.retokenize_v2 import _index_already_tokenized

    _make_done_chunk(tmp_path, "fleurs", "tam", "tam_00000007_deadbeef")
    assert _index_already_tokenized(str(tmp_path), "fleurs", "tam", 7)
    assert not _index_already_tokenized(str(tmp_path), "fleurs", "tam", 8)


def test_index_already_tokenized_returns_false_when_dir_missing(tmp_path):
    from scripts.retokenize_v2 import _index_already_tokenized
    assert not _index_already_tokenized(str(tmp_path), "fleurs", "kan", 0)
