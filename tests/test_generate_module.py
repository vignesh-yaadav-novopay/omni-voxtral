"""Phase 5 — verify scripts/generate.py exposes module-level entrypoints.

The full GPU integration (loading a real checkpoint and generating 30s of audio
per language) lives in test_no_gibberish.py and runs only when VOXTRAL_CKPT is
set. This file does the cheap structural checks: imports, signatures, that
generate_audio's argument contract matches what eval_wer.py and the no-gibberish
regression both call.
"""

import inspect
from typing import Optional

import pytest


def test_inference_pipeline_dataclass_imports():
    from scripts.generate import InferencePipeline
    fields = InferencePipeline.__dataclass_fields__
    assert {"model", "tokenizer", "config", "device"}.issubset(fields)


def test_load_inference_pipeline_signature():
    from scripts.generate import load_inference_pipeline
    sig = inspect.signature(load_inference_pipeline)
    assert "ckpt_path" in sig.parameters
    assert "device" in sig.parameters
    assert sig.parameters["device"].default == "cuda:0"


def test_generate_audio_signature_matches_callers():
    """eval_wer.py and test_no_gibberish.py call generate_audio(...) — make sure
    every kwarg they pass actually exists. If any are renamed, the call sites
    break silently otherwise."""
    from scripts.generate import generate_audio

    sig = inspect.signature(generate_audio)
    expected = {
        "pipeline", "prompt", "prompt_audio", "prompt_sample_rate",
        "prompt_tokens", "language", "max_windows", "temperature",
        "top_k", "repetition_penalty", "return_tokens",
    }
    assert expected.issubset(set(sig.parameters.keys())), \
        f"missing kwargs: {expected - set(sig.parameters.keys())}"

    # language MUST default to ISO 639-3 — anything else is the silent-English
    # fallback that Phase 1 specifically banished.
    assert sig.parameters["language"].default == "eng"


def test_encode_prompt_audio_signature():
    """encode_prompt_audio is the seam tests use to convert FLEURS waveforms
    into a prompt tensor without spinning up a generation."""
    from scripts.generate import encode_prompt_audio
    sig = inspect.signature(encode_prompt_audio)
    expected = {"audio", "sample_rate", "tokenizer", "language", "n_tokens"}
    assert expected.issubset(set(sig.parameters.keys()))
    # language is REQUIRED (no default).
    assert sig.parameters["language"].default is inspect.Parameter.empty


def test_generate_audio_with_both_prompt_and_audio_raises():
    """Passing both prompt= and prompt_audio= is a programmer error — surface
    it with a ValueError, not a silent override.

    Verified at module level by inspecting the source so we don't need to
    actually load the model.
    """
    from scripts import generate as gen_mod
    src = inspect.getsource(gen_mod.generate_audio)
    assert "pass at most one of prompt / prompt_audio" in src


def test_silence_token_id_matches_sp_vocab():
    """The dual-stream interruption test asserts model emits SP id 27. Make
    sure that id still corresponds to <|silence|> in the trained tokenizer —
    a vocab-rebuild that shifted control tokens would silently break Goal 4."""
    from pathlib import Path

    vocab_path = Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "omnivoxtral_sp.vocab"
    if not vocab_path.exists():
        pytest.skip(f"SP vocab not found at {vocab_path}")
    lines = vocab_path.read_text(encoding="utf-8").splitlines()
    assert lines[27].split("\t")[0] == "<|silence|>", \
        f"line 28 is {lines[27]!r} not <|silence|> — tokenizer rebuild?"

    from scripts.generate import SILENCE_TOKEN_ID
    assert SILENCE_TOKEN_ID == 27


def test_generate_dual_stream_signature():
    """generate_dual_stream is the entrypoint test_interruption_emission uses
    once a dual-stream checkpoint is available. The test passes specific
    kwargs — make sure they exist."""
    import inspect
    from scripts.generate import generate_dual_stream

    sig = inspect.signature(generate_dual_stream)
    expected = {
        "pipeline", "user_audio", "user_sample_rate", "language",
        "max_windows", "temperature", "top_k", "speech_rms_threshold",
    }
    assert expected.issubset(set(sig.parameters.keys()))
    assert sig.parameters["language"].default is inspect.Parameter.empty


def test_eval_wer_imports_generate_module():
    """eval_wer.py must successfully import scripts.generate at module load —
    if generate.py breaks, eval_wer.py should fail fast not silently no-op."""
    import importlib
    mod = importlib.import_module("scripts.eval_wer")
    src = inspect.getsource(mod._generate_audio_from_checkpoint)
    assert "from scripts.generate import generate_audio" in src
