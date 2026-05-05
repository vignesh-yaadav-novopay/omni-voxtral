"""MMS-1B-all wrapper for ASR fallback on the 7 Whisper-unsupported Indic langs.

Whisper-large-v3 supports 15/22 Indic. The remaining 7 (Bodo, Dogri, Konkani,
Manipuri, Santali, Sindhi, Kashmiri) are routed through `facebook/mms-1b-all`
with per-language adapters loaded at call time. Base model loaded once.

License: CC-BY-NC. Plan locks "research only" posture; sidecar
`data_license_class` is "cc-by-nc" for any tokens routed through this module.
"""

from __future__ import annotations

import typing

import numpy as np
import torch

# ISO 639-3 codes the SP tokenizer / sidecar use; MMS uses 3-letter codes too.
SUPPORTED_LANGUAGES: typing.ClassVar = {"brx", "doi", "kok", "mni", "sat", "snd", "kas"}


class MMSASR(torch.nn.Module):
    """Lazy-loaded wrapper around facebook/mms-1b-all.

    The base 1B model is loaded once; per-language adapters are switched via
    `processor.tokenizer.set_target_lang(lang)` and `model.load_adapter(lang)`
    on each `transcribe()` call. Fast-switching saves several GB of VRAM
    compared to loading per-language whole models.
    """

    SUPPORTED_LANGUAGES: typing.ClassVar[set[str]] = SUPPORTED_LANGUAGES

    def __init__(self, model_path: str = "facebook/mms-1b-all", device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self._processor = None
        self._model = None
        self._loaded_adapter: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_path).to(self.device)
        self._model.eval()  # set to inference mode

    def _ensure_adapter(self, language: str) -> None:
        if self._loaded_adapter == language:
            return
        self._processor.tokenizer.set_target_lang(language)
        self._model.load_adapter(language)
        self._loaded_adapter = language

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sample_rate: int, language: str) -> str:
        """audio: (1, num_samples) at 16 kHz mono.

        Raises ValueError if language not in SUPPORTED_LANGUAGES.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"MMSASR.transcribe: language={language!r} not in SUPPORTED_LANGUAGES "
                f"({sorted(SUPPORTED_LANGUAGES)}). Use Whisper for the others."
            )
        if sample_rate != 16_000:
            import torchaudio.functional as Faud
            audio = Faud.resample(audio, sample_rate, 16_000)
        self._ensure_loaded()
        self._ensure_adapter(language)
        # MMS expects float32 numpy
        if audio.dim() == 2:
            audio = audio[0]
        inputs = self._processor(
            audio.cpu().numpy(), sampling_rate=16_000, return_tensors="pt"
        ).to(self.device)
        logits = self._model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        transcript = self._processor.batch_decode(ids)[0]
        return transcript
