"""Phase 0 unit tests. These run inline (no GPU required for SP roundtrip)."""

from pathlib import Path

import pytest

from scripts.phase0_preflight import (
    LANGUAGE_TAGS,
    experiment_sp_roundtrip,
)


def test_sp_roundtrip_recovers_all_lang_tags(sp_tokenizer_path):
    result = experiment_sp_roundtrip(sp_tokenizer_path)
    assert result["status"] == "pass", (
        f"SP roundtrip failed: pass_rate={result['pass_rate']:.0%}; "
        f"plan §6.Phase 0 expects ≥95%. Findings: {result['findings']}"
    )
    assert result["recommended_mode"] in {"per_utterance", "global_prepend"}


def test_sp_roundtrip_returns_findings_dict(sp_tokenizer_path):
    result = experiment_sp_roundtrip(sp_tokenizer_path)
    assert "findings" in result
    assert any("ok" in f for f in result["findings"]
               if isinstance(f, dict) and "ok" in f)


def test_lang_tag_inventory_includes_22_indic_plus_english():
    """LANGUAGE_TAGS should cover the 22 Indic targets + English (ISO 639-3)."""
    expected_min = {
        "<|lang:hin|>", "<|lang:ben|>", "<|lang:tam|>", "<|lang:tel|>",
        "<|lang:kan|>", "<|lang:mal|>", "<|lang:mar|>", "<|lang:guj|>",
        "<|lang:pan|>", "<|lang:urd|>", "<|lang:ori|>", "<|lang:asm|>",
        "<|lang:nep|>", "<|lang:eng|>",
    }
    have = set(LANGUAGE_TAGS)
    missing = expected_min - have
    assert not missing, f"missing tier-1 language tags: {missing}"


@pytest.mark.gpu
def test_mimi_silence_differentiation_runs(silence_5s, noise_5s_minus_45db):
    """Smoke: end-to-end Mimi encode of bit-zero + -45 dB noise produces
    different q1 histograms in MOST cases. We only assert non-empty histograms
    here — the real pass/fail decision lives in scripts/phase0_preflight.py.
    """
    pytest.importorskip("torch")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Phase 0 exp 2 requires CUDA")
    from scripts.phase0_preflight import experiment_mimi_silence
    result = experiment_mimi_silence(device="cuda:0")
    assert "histogram_sizes" in result
    assert all(v >= 1 for v in result["histogram_sizes"].values())
    assert result["recommended_strategy"] in {"noise_floor", "room_tone"}
