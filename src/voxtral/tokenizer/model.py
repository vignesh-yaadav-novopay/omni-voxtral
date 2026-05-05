"""VoxtralTokenizer — single-stream Mimi + Whisper encoder.

v2 changes (Phase 1 of the rebuild):
- `VoxtralTokenizerConfig.language` defaults to None (was "en"). Per-call language
  always wins, but if config and call both omit it, encode/translate raise.
- `VoxtralTokenizerConfig.sp_tokenizer_path` defaults to the trained SentencePiece
  Unigram model (was None, which silently fell back to Mistral BPE — Defect 2).
- `encode()` now returns `(tokens, metadata)`. The metadata dict is the partial
  sidecar payload; callers merge in `source_metadata` and Phase 3 LID/diarization
  fields, then write via `write_metadata_sidecar`.
- New `translate()` method runs Whisper with `task="translate"` and returns the
  English string only — for sidecar-only English translations.
"""

import os
import typing

import dotenv
import huggingface_hub as hf_hub
import torch
import torchaudio as ta

from .mimi.models import loaders
from .word_level_whisper import TimedWhisperTokenizer

dotenv.load_dotenv()


_DEFAULT_SP_MODEL = "data/tokenizer/omnivoxtral_sp.model"


# ISO 639-1 → ISO 639-3. ONLY languages that have real ISO 639-1 codes go
# in this map. The 7 MMS-routed Indic languages (brx, doi, kok, mni, sat,
# snd, kas) and a few others without 639-1 codes are listed separately in
# `_ISO3_ONLY` below. The split matters: iso3_to_whisper_code() must return
# None for MMS-only langs so the encode path knows to route them through MMS
# instead of crashing in Whisper.
_ISO1_TO_ISO3 = {
    "as": "asm", "bn": "ben", "en": "eng", "gu": "guj", "hi": "hin",
    "kn": "kan", "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori",
    "pa": "pan", "ta": "tam", "te": "tel", "ur": "urd",
}
_ISO3_ONLY = {
    "brx", "doi", "kas", "kok", "mai", "mni", "san", "sat", "snd",
}
_ALL_ISO3 = set(_ISO1_TO_ISO3.values()) | _ISO3_ONLY


_ISO3_TO_ISO1 = {v: k for k, v in _ISO1_TO_ISO3.items()}


# Aliases the wild produces that aren't strict ISO 639-3:
# - MMS LID emits "npi" (Western Nepali, ISO 639-3) → map to "nep" (macrolanguage we use)
# - Whisper LID emits "ory" for Odia (ISO 639-2/T) → map to "ori" (ISO 639-2/B form we use)
# - "sin" sometimes seen for Sindhi → map to "snd"
# - Two-letter "or" / "ne" (ISO 639-1) handled by _ISO1_TO_ISO3 above.
_LANG_ALIASES = {
    "npi": "nep",   # Nepali (Western) → Nepali macrolanguage
    "ory": "ori",   # Odia (B-form code variant)
    "sin": "snd",   # Sindhi sometimes seen
    "bod": "brx",   # Bodo (rare alias)
}


def normalize_language_to_iso3(lang: str) -> str:
    """Normalize ISO 639-1 / ISO 639-3 / known aliases to our canonical 639-3.

    Idempotent for already-canonical inputs. Raises on unknown languages.
    """
    if not lang:
        raise ValueError("normalize_language_to_iso3: empty language")
    if lang in _ISO1_TO_ISO3:
        return _ISO1_TO_ISO3[lang]
    if lang in _LANG_ALIASES:
        return _LANG_ALIASES[lang]
    if lang in _ALL_ISO3:
        return lang
    raise ValueError(
        f"Unknown language code {lang!r}; expected ISO 639-1 (e.g. 'hi') or "
        f"ISO 639-3 (e.g. 'hin'). Known: {sorted(_ALL_ISO3)}."
    )


def iso3_to_whisper_code(lang_iso3: str) -> str | None:
    """Map ISO 639-3 to the ISO 639-1 form Whisper accepts.

    Returns None for the 7 MMS-only Indic languages (brx, doi, kok, mni, sat,
    snd, kas) plus a few others without Whisper coverage (mai, san). The
    encode path must check for None and route those calls through MMS.
    """
    return _ISO3_TO_ISO1.get(lang_iso3)


class VoxtralTokenizerConfig(typing.NamedTuple):
    mimi_path: str = "kyutai/mimi"
    whisper_path: str = "openai/whisper-large-v3"
    text_hz: int = 5
    mimi_num_quantizers: int = 8
    text_vocab_size: int = 65536
    # CHANGED (Phase 1): None means caller MUST pass language at every encode/translate.
    # Defect 1 fix.
    language: str | None = None
    # CHANGED (Phase 1): defaults to the trained SentencePiece model. Mistral BPE
    # remains opt-in via explicit None. Defect 2 fix.
    sp_tokenizer_path: str | None = _DEFAULT_SP_MODEL


def interleave(*seqs: list[torch.Tensor], factors: list[int]):
    assert isinstance(seqs[0], torch.Tensor)
    bs = seqs[0].size(0)

    to_cat = []
    for i, seq in enumerate(seqs):
        assert isinstance(seq, torch.Tensor)
        seq = seq.view(bs, -1, factors[i])
        to_cat.append(seq)

    out = torch.cat(to_cat, dim=-1)
    return out.view(bs, -1)


def uninterleave(x: torch.Tensor, factors: list[int]) -> list[torch.Tensor]:
    bs = x.size(0)
    chunks = x.view(bs, -1, sum(factors))
    splits = chunks.split(factors, dim=-1)

    return [split.reshape(bs, -1) for split in splits]


class VoxtralTokenizer(torch.nn.Module):
    def __init__(self, config: VoxtralTokenizerConfig):
        super().__init__()
        self.config = config

        # mimi
        mimi_weight = hf_hub.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device="cpu")
        self.mimi.set_num_codebooks(config.mimi_num_quantizers)

        self.text_to_audio_token_factor = int(
            self.mimi.frame_rate * self.config.mimi_num_quantizers / self.config.text_hz
        )
        self.stream_stride = 1 + self.text_to_audio_token_factor  # 21 for default config

        # whisper
        self.whisper = TimedWhisperTokenizer(
            config.whisper_path,
            hertz=config.text_hz,
            language=config.language,
            sp_tokenizer_path=config.sp_tokenizer_path,
        )

    @property
    def device(self) -> torch.device:
        return next(self.mimi.parameters()).device

    def _resolve_language(self, language: str | None) -> str:
        """Resolve + normalize the language to ISO 639-3.

        The SP vocab uses 639-3 (e.g. <|lang:hin|>), but callers commonly pass
        639-1 from dataset metadata (e.g. FLEURS_LANGS keys are "hi", "ta").
        Normalize once here so downstream code doesn't have to.
        """
        effective = language if language is not None else self.config.language
        if effective is None:
            raise ValueError(
                "VoxtralTokenizer: `language` must be passed at config or per-call. "
                "Defect 1 fix: no silent fallback to 'en'."
            )
        return normalize_language_to_iso3(effective)

    def _build_metadata(
        self,
        x: torch.Tensor,
        sample_rate: int,
        tokens: torch.Tensor,
        whisper_meta: dict[str, typing.Any],
        language: str,
        task: str,
        source_metadata: dict | None,
    ) -> dict:
        per_batch = whisper_meta.get("per_batch") or [{}]
        first = per_batch[0] if per_batch else {}
        num_samples = int(x.size(-1))
        meta: dict[str, typing.Any] = {
            "schema_version": 2,
            "preprocessing_version": "v2.0",
            # `language` is always written in ISO 639-3 (e.g. "hin") so the
            # trainer's language-temperature sampler / val-split builder don't
            # have to mix conventions.
            "language": language,
            "language_method": "caller_provided",
            "language_confidence": 1.0,
            "transcript": first.get("transcript", ""),
            "translation_en": None,
            "transcript_method": "whisper_large_v3",
            "task": task,
            "duration_s": float(num_samples / sample_rate),
            "sample_rate": int(sample_rate),
            "num_channels": int(x.size(1)),
            "num_speakers": 1,
            "speaker_segments": [],
            "stream_layout": "single",
            "tokenizer_config": {
                "sp_model": (
                    os.path.basename(self.config.sp_tokenizer_path)
                    if self.config.sp_tokenizer_path
                    else None
                ),
                "text_hz": self.config.text_hz,
                "mimi_num_quantizers": self.config.mimi_num_quantizers,
                "stride": self.stream_stride,
            },
            "token_count": int(tokens.size(-1)),
            "token_range": [int(tokens.min().item()), int(tokens.max().item())],
            "valid_token_mask": None,  # Phase 1.5 fills this in via writer helper
        }
        if source_metadata:
            meta.update(source_metadata)
        return meta

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        sample_rate: int,
        language: str | None = None,
        task: str = "transcribe",
        source_metadata: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Encode audio to interleaved tokens + sidecar metadata.

        Args:
            x: (batch, 1, num_samples) float at sample_rate Hz, mono. Mimi is mono only.
            sample_rate: must be 24000.
            language: ISO code. Per-call value wins over config.language.
            task: "transcribe" (source-language text) or "translate" (English).
            source_metadata: dict merged into the returned metadata (source, source_id,
                source_url, chunk_index, …).

        Returns:
            tokens: (batch, seq_len) int64.
            metadata: dict with the v2 sidecar payload for batch[0]. For batch>1 the
                caller is responsible for splitting per-item metadata, but we always
                fill the dict from batch[0]'s transcript — this is consistent with the
                preprocessing pipeline always running batch=1.
        """
        assert x.dim() == 3, f"audio must be (B, 1, T); got {tuple(x.shape)}"
        assert x.size(1) == 1, "Mimi is mono — pass mono audio (or split stereo upstream)"
        assert sample_rate == 24_000, f"sample_rate must be 24000, got {sample_rate}"
        effective_lang = self._resolve_language(language)

        x_for_whisper = ta.functional.resample(x, sample_rate, 16_000)[:, 0]
        # effective_lang is ISO 639-3; Whisper wants ISO 639-1.
        whisper_lang = iso3_to_whisper_code(effective_lang)
        if whisper_lang is None:
            raise ValueError(
                f"Whisper does not support language {effective_lang!r}. The 7 "
                "Indic languages without Whisper coverage (brx, doi, kok, mni, "
                "sat, snd, kas) must route through MMS — see scripts/detect_language.py"
            )
        text_tokens, whisper_meta = self.whisper.forward_with_transcript(
            x_for_whisper, sample_rate=16_000,
            language=whisper_lang, task=task,
            sp_lang_iso3=effective_lang,
        )

        # make sure every quantizer uses different tokens
        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = (
            torch.arange(0, self.config.mimi_num_quantizers) * mimi_vocab_size
        )

        mimi_tokens = self.mimi.encode(x.to(self.device))

        audio_tokens = (
            mimi_tokens
            + token_offset[None, :, None].to(self.device)
            + self.config.text_vocab_size
        )

        interleaved_audio_tokens = interleave(
            *audio_tokens.unbind(1), factors=[1] * self.config.mimi_num_quantizers
        )

        # delay text and audio tokens by two windows
        # the text is not perfectly 5 hz aligned, so we avoid corresponding audio
        # tokens appearing before the text tokens by accident.
        # we remove the first two windows of the audio tokens
        # and the last two windows of the text tokens
        interleaved_audio_tokens = interleaved_audio_tokens[
            ..., int(self.config.mimi_num_quantizers * self.mimi.frame_rate * 2):
        ]

        text_tokens = text_tokens[..., : -self.config.text_hz * 2]

        intermediate_tokens = [text_tokens, interleaved_audio_tokens]
        tokens = interleave(
            *intermediate_tokens, factors=[1, int(self.text_to_audio_token_factor)]
        )

        metadata = self._build_metadata(
            x, sample_rate, tokens, whisper_meta,
            language=effective_lang, task=task, source_metadata=source_metadata,
        )
        return tokens, metadata

    @torch.no_grad()
    def translate(
        self,
        x: torch.Tensor,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        """Run Whisper with task='translate', return English transcript only.

        No Mimi pass — this is the cheap second pass for sidecar `translation_en`.
        Output capped at 2000 chars (defensive).
        """
        assert x.dim() == 3 and x.size(1) == 1
        assert sample_rate == 24_000
        effective_lang = self._resolve_language(language)  # ISO 639-3
        whisper_lang = iso3_to_whisper_code(effective_lang)
        if whisper_lang is None:
            raise ValueError(
                f"Whisper translate doesn't support {effective_lang!r}; route MMS langs separately."
            )
        x_for_whisper = ta.functional.resample(x, sample_rate, 16_000)[:, 0]
        return self.whisper.translate(
            x_for_whisper, sample_rate=16_000, language=whisper_lang
        )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2
        # uninterleave tokens to separate text and audio tokens; throw away text
        _, audio_tokens = uninterleave(z, factors=[1, self.text_to_audio_token_factor])

        audio_tokens = audio_tokens - self.config.text_vocab_size

        audio_tokens = uninterleave(
            audio_tokens, factors=[1] * self.config.mimi_num_quantizers
        )

        audio_tokens = torch.stack(audio_tokens, dim=1)

        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = (
            torch.arange(0, self.config.mimi_num_quantizers, device=audio_tokens.device)
            * mimi_vocab_size
        )
        audio_tokens = audio_tokens - token_offset[None, :, None]
        # Clamp out-of-range tokens (common with small/undertrained models)
        audio_tokens = torch.clamp(
            audio_tokens, min=0, max=self.mimi.quantizer.cardinality - 1
        )
        decoded_audio = self.mimi.decode(audio_tokens)

        return decoded_audio
