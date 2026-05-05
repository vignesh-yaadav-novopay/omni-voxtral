"""Phase 3 — LID + MMS routing tests.

Heavy LID stack (mms-lid-2048 ~4GB, faster-whisper ~3GB) is GPU-only and
gated behind @pytest.mark.gpu. Pure logic tests run inline.
"""

import pytest

from src.voxtral.tokenizer.mms_asr import SUPPORTED_LANGUAGES as MMS_LANGS
from scripts.detect_language import (
    ISO1_TO_ISO3,
    _normalize_lang,
)


def test_iso_normalization_round_trip():
    """Whisper returns 'hi'; we normalize to 'hin' to match SP and MMS."""
    assert _normalize_lang("hi") == "hin"
    assert _normalize_lang("ta") == "tam"
    assert _normalize_lang("eng") == "eng"  # already ISO3


def test_mms_languages_covers_seven_whisper_unsupported():
    """Plan §4 / LLD §3.3 — exactly these 7 must be MMS-routed."""
    expected = {"brx", "doi", "kok", "mni", "sat", "snd", "kas"}
    assert MMS_LANGS == expected


def test_mms_iso3_codes_disjoint_from_whisper_supported():
    """No language should be routed through both Whisper and MMS."""
    whisper_supported = set(ISO1_TO_ISO3.values())
    overlap = MMS_LANGS & whisper_supported
    assert overlap == set(), f"unexpected overlap: {overlap}"


def test_mms_asr_rejects_unsupported_language(tmp_path):
    """MMSASR.transcribe with a non-MMS language raises a clear error."""
    import torch
    from src.voxtral.tokenizer.mms_asr import MMSASR

    asr = MMSASR(device="cpu")  # don't load model in CI test path
    audio = torch.zeros(1, 16_000)
    with pytest.raises(ValueError, match="not in SUPPORTED_LANGUAGES"):
        asr.transcribe(audio, sample_rate=16_000, language="hin")
