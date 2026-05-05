"""Phase 1.5 — verify build_val_split.py picks deterministic, stratified val.

The trainer reads `data/val_split_v2.json` to pin val files across phases; if
this script's output drifts (e.g. picks different files on different runs),
val-loss comparisons across phases become meaningless.
"""

import json

import numpy as np
import pytest

from voxtral.data.sidecar import atomic_save_npy, write_metadata_sidecar


def _make_token_corpus(tmp_path, languages, n_per_lang=20):
    """Lay down a fake v2 token tree under `tmp_path/<source>/<lang>/<shard>/*.npy`
    with paired sidecars. Mirrors what retokenize_v2.py would have produced."""
    for lang in languages:
        for i in range(n_per_lang):
            shard = f"{lang}_{i:08d}"[:2]
            d = tmp_path / "fleurs" / lang / shard
            d.mkdir(parents=True, exist_ok=True)
            npy = d / f"{lang}_{i:08d}_hash.npy"
            atomic_save_npy(np.arange(50, dtype=np.int64), str(npy))
            write_metadata_sidecar(
                str(npy),
                {"language": lang, "source": "fleurs", "duration_s": 12.0,
                 "stream_layout": "single", "tokenizer_config": {"stride": 21}},
            )
    return tmp_path


def test_discover_files_groups_by_sidecar_language(tmp_path):
    from scripts.build_val_split import discover_files_by_language

    _make_token_corpus(tmp_path, ["hin", "ben", "tam"], n_per_lang=10)
    by_lang = discover_files_by_language(str(tmp_path))
    assert set(by_lang.keys()) == {"hin", "ben", "tam"}
    assert all(len(files) == 10 for files in by_lang.values())


def test_discover_skips_files_without_sidecar(tmp_path):
    """Files without a sidecar must NOT be sampled — the trainer can't read
    their language, so they're not legal val candidates."""
    from scripts.build_val_split import discover_files_by_language

    d = tmp_path / "fleurs" / "hin" / "hi"
    d.mkdir(parents=True)
    atomic_save_npy(np.arange(10, dtype=np.int64), str(d / "orphan.npy"))
    by_lang = discover_files_by_language(str(tmp_path))
    assert by_lang == {}


def test_stratified_sample_is_deterministic_across_calls(tmp_path):
    """Two calls with the same seed → identical val sets. This is the whole
    point of pinning."""
    from scripts.build_val_split import discover_files_by_language, stratified_sample

    _make_token_corpus(tmp_path, ["hin", "ben"], n_per_lang=20)
    by_lang = discover_files_by_language(str(tmp_path))

    sel_a, counts_a = stratified_sample(by_lang, per_lang=5, seed=42)
    sel_b, counts_b = stratified_sample(by_lang, per_lang=5, seed=42)
    assert sel_a == sel_b
    assert counts_a == counts_b == {"hin": 5, "ben": 5}


def test_stratified_sample_different_seed_picks_different_files(tmp_path):
    from scripts.build_val_split import discover_files_by_language, stratified_sample

    _make_token_corpus(tmp_path, ["hin"], n_per_lang=30)
    by_lang = discover_files_by_language(str(tmp_path))

    sel_a, _ = stratified_sample(by_lang, per_lang=5, seed=1)
    sel_b, _ = stratified_sample(by_lang, per_lang=5, seed=2)
    assert sel_a != sel_b


def test_stratified_sample_takes_all_when_short(tmp_path):
    """If a language has fewer files than per_lang, take everything available."""
    from scripts.build_val_split import discover_files_by_language, stratified_sample

    _make_token_corpus(tmp_path, ["doi"], n_per_lang=3)  # severely underrepresented
    by_lang = discover_files_by_language(str(tmp_path))

    sel, counts = stratified_sample(by_lang, per_lang=30, seed=42)
    assert counts["doi"] == 3
    assert len(sel) == 3


def test_write_val_split_emits_v2_schema(tmp_path):
    from scripts.build_val_split import write_val_split, SCHEMA_VERSION

    files = [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")]
    counts = {"hin": 1, "ben": 1}
    out = tmp_path / "val_split_v2.json"
    write_val_split(str(out), files, counts, seed=1729)

    payload = json.loads(out.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["seed"] == 1729
    assert payload["files"] == files
    assert payload["per_language"] == counts
    assert "created_at" in payload


def test_write_val_split_uses_atomic_replace(tmp_path):
    """No .tmp file should remain after a successful write — atomicity matters
    when concurrent jobs might glance at val_split_v2.json mid-write."""
    from scripts.build_val_split import write_val_split

    out = tmp_path / "data" / "val_split_v2.json"
    write_val_split(str(out), [], {}, seed=1)
    assert out.exists()
    assert not (out.parent / "val_split_v2.json.tmp").exists()


def test_val_split_is_consumable_by_trainer(tmp_path):
    """End-to-end: build_val_split → trainer reads it → train/val are disjoint."""
    from scripts.build_val_split import (
        discover_files_by_language, stratified_sample, write_val_split,
    )
    from voxtral.trainer.config import VoxtralTrainConfig
    from voxtral.trainer.data import VoxtralDataset

    _make_token_corpus(tmp_path, ["hin", "ben"], n_per_lang=10)
    by_lang = discover_files_by_language(str(tmp_path))
    sel, counts = stratified_sample(by_lang, per_lang=3, seed=42)
    pin_path = tmp_path / "val_split_v2.json"
    write_val_split(str(pin_path), sel, counts, seed=42)

    cfg = VoxtralTrainConfig(fake=False, data_path=str(tmp_path), max_steps=1)
    val_ds = VoxtralDataset(cfg, split="val", val_split_path=str(pin_path))
    train_ds = VoxtralDataset(cfg, split="train", val_split_path=str(pin_path))

    assert sorted(val_ds.file_paths) == sorted(sel)
    for f in sel:
        assert f not in train_ds.file_paths
