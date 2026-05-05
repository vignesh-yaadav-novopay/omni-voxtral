# OmniVoxtral Pipeline Rebuild — Low-Level Design

## 1. Purpose and scope

This document specifies file-by-file changes, function signatures, data schemas, and runtime contracts for the rebuild defined in `plan.md` and architected in `HLD.md`. The reader is the engineer (or `deep-implement` agent) who writes the code. Every signature here must match the implementation exactly — no improvising.

## 2. Conventions

- All paths are relative to `/apps/voxtral` unless absolute.
- Python ≥3.11. `uv run` is the runner. Type hints on every public function.
- Tensors documented as `(batch, channels, samples)` for audio; `(batch, seq_len)` for tokens.
- File I/O is atomic: write to `<path>.tmp`, then `os.replace(<path>.tmp, <path>)`.
- All preprocessing scripts launch via `bash autoresearch/scripts/run_safe.sh "..." <budget_seconds>`.

## 3. Module-by-module specification

### 3.1 `src/voxtral/tokenizer/model.py`

#### `VoxtralTokenizerConfig`

```python
class VoxtralTokenizerConfig(NamedTuple):
    mimi_path: str = "kyutai/mimi"
    whisper_path: str = "openai/whisper-large-v3"
    text_hz: int = 5
    mimi_num_quantizers: int = 8
    text_vocab_size: int = 65536
    language: str | None = None                                  # CHANGED: was "en"; now None means "must be passed per-call"
    sp_tokenizer_path: str | None = "data/tokenizer/omnivoxtral_sp.model"  # CHANGED: was None; now defaults to SentencePiece
```

Migration: any caller relying on `language="en"` default must explicitly pass `language="en"`.

#### `VoxtralTokenizer.encode`

```python
@torch.no_grad()
def encode(
    self,
    x: torch.Tensor,
    sample_rate: int,
    language: str | None = None,
    task: str = "transcribe",
    source_metadata: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    """Encode audio to interleaved tokens + metadata.

    Args:
        x: (batch, 1, num_samples) float at sample_rate Hz, mono.
        sample_rate: must be 24000.
        language: ISO code. If None, reads from config.language. If both None, raises.
        task: "transcribe" (source language) or "translate" (English).
        source_metadata: dict merged into sidecar (source, source_id, source_url, chunk_index).

    Returns:
        tokens: (batch, seq_len) int64.
        metadata: dict with all fields needed for the sidecar (see Section 4 schema).

    Raises:
        ValueError: if language is None in both call and config.
        AssertionError: if x.dim() != 3, x.size(1) != 1, or sample_rate != 24000.
    """
```

#### `VoxtralTokenizer.translate`

```python
@torch.no_grad()
def translate(
    self,
    x: torch.Tensor,
    sample_rate: int,
    language: str | None = None,
) -> str:
    """Run Whisper with task='translate', return English transcript only.

    Returns English translation as plain string. Truncated to 2000 chars with
    logged warning if longer.
    """
```

### 3.2 `src/voxtral/tokenizer/word_level_whisper.py`

`TimedWhisperTokenizer.__init__` signature: `language: str | None = None` (was `language: str = "en"`). Per-call language wins; init default is fallback only.

`TimedWhisperTokenizer.forward` signature:

```python
def forward(
    self,
    audio: torch.Tensor,
    sample_rate: int,
    language: str | None = None,
    task: str = "transcribe",
) -> torch.Tensor:
    """Bucketed text tokens via Whisper.

    audio: (batch, num_samples) at 16 kHz, mono.
    Returns: (batch, num_buckets) where each bucket is exactly self.hertz tokens.

    Raises ValueError if language is None at both call and init.
    """
```

`generate_tokens` (module-level helper):

```python
def generate_tokens(
    processor: Any,
    model: Any,
    audio: torch.Tensor,
    language: str,                           # REQUIRED (was default "en")
    task: str = "transcribe",                # NEW
) -> dict[str, Any]:
    """Always forward language and task to model.generate(). No silent auto-detect."""
```

The previous `if language != "en"` branch (lines 177-180) is **deleted**.

`TimedWhisperTokenizer.translate` (new method):

```python
@torch.no_grad()
def translate(
    self,
    audio: torch.Tensor,
    sample_rate: int,
    language: str | None = None,
) -> str:
    """Single-pass translate. Output capped at 2000 chars; truncate + log if exceeded."""
```

### 3.3 `src/voxtral/tokenizer/mms_asr.py` (new file)

```python
class MMSASR(nn.Module):
    """Wrapper around facebook/mms-1b-all for ASR fallback on the 7 Indic languages
    Whisper-large-v3 doesn't support: brx, doi, kok, mni, sat, sd, ks.

    Loads the 1B base model once. Switches per-language adapters at call time."""

    SUPPORTED_LANGUAGES: ClassVar[set[str]] = {"brx", "doi", "kok", "mni", "sat", "sd", "ks"}

    def __init__(self, model_path: str = "facebook/mms-1b-all", device: str = "cuda"):
        ...

    @torch.no_grad()
    def transcribe(self, audio: torch.Tensor, sample_rate: int, language: str) -> str:
        """audio: (batch=1, num_samples) at 16 kHz mono.
        language: one of SUPPORTED_LANGUAGES.
        Raises ValueError if language not in SUPPORTED_LANGUAGES.
        """
```

### 3.4 `src/voxtral/tokenizer/dual_stream.py`

`DualStreamTokenizer.encode_with_metadata` (new):

```python
@torch.no_grad()
def encode_with_metadata(
    self,
    user_audio: torch.Tensor,
    model_audio: torch.Tensor,
    sample_rate: int,
    segments_metadata: list[dict],                     # NEW: from diarize_v2.py
    language: str,
    user_translation_en: str | None = None,
    model_translation_en: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """Encode dual streams + return metadata for sidecar.
    Returns: (tokens with stride-42 layout, metadata dict).
    """
```

#### Silence handling

The non-active speaker's audio is masked **before** Mimi encode. The mask source is determined by Phase 0 experiment 2:
- `silence_strategy="noise_floor"` → -45 dB Gaussian noise
- `silence_strategy="room_tone"` → looped 200 ms sample of low-energy regions of the same source video

Implementation: `_apply_silence_mask(audio, segments, target_speaker, strategy)` helper, called before each `_encode_audio_tokens`.

### 3.5 `src/voxtral/data/preprocessing.py`

`AudioChunkDataset` — variable-length support:

```python
class AudioChunkDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        target_sample_rate: int,
        num_channels: int,
        dtype: torch.dtype,
        chunk_frames: int | None = None,                # CHANGED: now optional
    ):
        ...

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        """Returns (waveform, filename) where waveform shape is (num_channels, num_samples)
        — variable length. NO PADDING. NO TRUNCATION.
        If self.chunk_frames is set (legacy path for Phase 1a), pad/truncate as before."""
```

`_save_tokens` and new `write_metadata_sidecar` — both atomic via `os.replace`.

### 3.6 `src/voxtral/trainer/data.py`

`VoxtralDataset.__init__` adds:

```python
val_split_path: str | None = "data/val_split_v2.json",
require_sidecar: bool = True,
sidecar_missing_threshold: float = 0.05,
```

Internally: `self._rng = random.Random(config.seed)` (local instance, not global state).

If `val_split_path` exists, loads pinned validation file list; otherwise falls back to deterministic seed-shuffle.

Yielded dict per sample:

```python
{
    "tokens": Tensor[(seq_len,)],
    "language": str,
    "duration_s": float,
    "source": str,
    "valid_token_mask": Tensor[(seq_len,)],     # bool, all-True if missing
    "stream_layout": str,                       # "single" | "dual"
}
```

After every 1000 yields, if `_sidecar_missing / total > 0.05`, raise `RuntimeError`.

#### Stride-homogeneous sampling

Implemented in `VoxtralDataset._make_sampler()`. Pre-scan files. Group by sidecar `stream_layout`. Each batch is drawn from a single group. DDP-aware: per-rank file allocation respects group membership.

#### Temperature sampling τ=3.3

```python
def _compute_language_weights(self) -> dict[str, float]:
    """π_l = ω_l^(1/τ) / Σ ω_l'^(1/τ), where ω_l = file count per language. τ=3.3."""
```

Sampling weighted by `language_weights[sidecar.language]`.

### 3.7 `src/voxtral/trainer/omni_trainer.py`

`compute_omni_loss` adds `valid_token_mask` argument:

```python
def compute_omni_loss(
    model: ...,
    x: torch.Tensor,                                   # (batch, seq_len)
    config: VoxtralTrainConfig,
    valid_token_mask: torch.Tensor | None = None,      # NEW (batch, seq_len) bool
) -> dict[str, torch.Tensor]:
    ...
    # After CE computation:
    if valid_token_mask is not None:
        token_mask = valid_token_mask[:, 1:]  # align with shift-by-1
        temporal_loss_per_pos = temporal_loss_per_pos * token_mask.float()
        temporal_loss = temporal_loss_per_pos.sum() / token_mask.sum().clamp_min(1)
```

Same masking applied to `depth_loss` for any frame whose midpoint falls in a masked region.

#### Mean-across-ranks val (audit AF-406)

```python
def _aggregate_val_metrics(local_metrics: dict[str, float]) -> dict[str, float]:
    """All-reduce mean across ranks. Also compute std across ranks."""
    if not torch.distributed.is_initialized():
        return {**local_metrics, **{f"{k}_std": 0.0 for k in local_metrics}}
    # ... gather + mean + std
```

### 3.8 `scripts/phase0_preflight.py` (new)

Runs all 3 (formerly 4) experiments in sequence. Outputs `planning/phase0_results.md`.

```python
def run_phase0_preflight(planning_dir: str = "planning") -> dict:
    """Returns:
        {
            "sp_roundtrip": {"status": "pass"|"fail", "details": str},
            "mimi_silence_diff": {"status": "pass"|"fail", "strategy": "noise_floor"|"room_tone"},
            "lid_pilot": {
                "agreement_rate": float,
                "low_confidence_rate": float,
                "total_quarantine_rate": float,
                "recommended_threshold": float,
            }
        }
    Side effects: writes <planning_dir>/phase0_results.md.
    """
```

Each sub-experiment is a function. Each writes a section to the report. Each is unit-testable from `tests/test_phase0_preflight.py`.

### 3.9 `scripts/retokenize_v2.py` (new — Phase 1 driver)

```python
def retokenize_dataset(
    dataset: str,                # "fleurs" | "indicvoices" | "youtube"
    languages: list[str],
    input_path: str | None,      # for youtube; None for fleurs/iv
    output_dir: str = "data/tokens_v2",
    translation_sample_rate: float = 0.10,
    device: str = "cuda:0",
    max_files_per_lang: int | None = None,
):
    """For each language → tokenize via VoxtralTokenizer.encode → write .npy + .meta.json.
    Writes translation sidecar for translation_sample_rate fraction of files.
    """
```

Usage:
- `uv run scripts/retokenize_v2.py --dataset fleurs --languages hi,ta,bn,...`
- `uv run scripts/retokenize_v2.py --dataset youtube --languages all --input_path data/chunks_indic_yt`

### 3.10 `scripts/vad_chunker.py` (new — Phase 2 driver)

```python
def chunk_audio_file(
    input_path: str,
    output_dir: str,
    source: str,                                         # "yt" | "iv" | ...
    language_or_unknown: str,
    silero_params: dict,
    detect_stereo_speakers: bool = True,
    correlation_threshold: float = 0.5,
) -> list[dict]:
    """Returns list of chunk metadata dicts (one per chunk emitted)."""

def run_silero_vad_per_channel(
    waveform: torch.Tensor,                              # (num_channels, num_samples)
    sample_rate: int,
    params: dict,
) -> list[list[tuple[float, float]]]:
    """Returns per-channel speech boundaries [(start_s, end_s), ...]."""

def detect_speaker_per_channel(stereo_waveform: torch.Tensor) -> bool:
    """Compute Pearson correlation between L/R. Return True if correlation < 0.5."""
```

Output `chunk.json` per chunk: `source`, `source_id`, `source_url`, `chunk_index`, `start_s_in_source`, `end_s_in_source`, `duration_s`, `num_channels`, `stream_role` ("single" | "user" | "model"), `lid_inherits_from_neighbor`.

### 3.11 `scripts/detect_language.py` (new — Phase 3 driver)

```python
def detect_chunk_language(
    chunk_path: str,
    primary_lid_model: Any,                              # mms-lid-2048 instance
    secondary_lid_model: Any,                            # whisper-large-v3 instance
    confidence_threshold: float = 0.7,
    streaming_context_threshold_s: float = 4.0,
    neighbor_chunks: list[str] | None = None,
) -> dict:
    """Returns {language, confidence, method, secondary, agreement}.
    For chunks shorter than streaming_context_threshold_s, prepends/appends ±2s
    neighbor audio for LID context. The transcription pass that follows uses the
    SAME context.
    """

def batched_lid_pass(chunks_dir: str, output: str = ".lang.json"):
    """First pass: load mms-lid-2048, sweep all chunks, unload, write .lang.json."""

def batched_asr_pass(chunks_dir: str):
    """Second pass: load Whisper-large-v3 OR MMSASR depending on language route,
    sweep all chunks, write transcript directly to memory or to a transient buffer
    that retokenize_v2.py reads."""
```

### 3.12 `scripts/detect_music.py` (new — Phase 4 helper)

```python
def detect_music_in_source(audio_path: str) -> dict:
    """Returns {music_likely: bool, harmonicity: float, tempo_detected: bool, ...}.
    Computes: harmonicity ratio (librosa), spectral contrast, beat-track success.
    Heuristic: harmonicity > 0.6 AND tempo detected → music_likely=True.
    """
```

Run once over `data/indic_urls.txt`, write `data/source_music_flags.json`.

### 3.13 `scripts/diarize_v2.py` (new — Phase 4 driver)

Replaces `scripts/diarize_audio.py` (legacy retained but unused).

```python
def diarize_source(
    source_audio_path: str,
    output_dir: str,
    use_demucs: bool,                                    # gated by music_likely
    silence_strategy: str,                               # "noise_floor" | "room_tone" (Phase 0 result)
    device: str = "cuda:0",
) -> list[dict]:
    """Pipeline: (Demucs?) → faster-whisper → ctc-forced-aligner → Silero VAD →
    pyannote-3.1 → speaker labels merged onto words.

    Output structure: data/dual_chunks_v2/{source}/{lang}/{shard}/{uuid}.json
    plus S0.wav and S1.wav (other speaker masked per silence_strategy).
    """
```

### 3.14 `scripts/eval_wer.py` (new — Phase 5 evaluation)

```python
def run_wer_evaluation(
    checkpoint_path: str,
    languages: list[str],                                # default: 13 FLEURS langs
    samples_per_language: int = 5,
    whisper_model: str = "openai/whisper-large-v3",
    output_path: str = "logs/eval/<run_id>/wer.json",
) -> dict:
    """Generate samples, transcribe with Whisper, compute WER per language.
    Returns aggregate report. Fails the run (exit 1) if any language WER > 80%."""
```

## 4. Sidecar metadata schema (JSON)

### 4.1 Schema v2

```json
{
  "schema_version": 2,
  "preprocessing_version": "v2.0",
  "preprocessing_run_id": "uuid-string",
  "preprocessing_timestamp": "2026-05-05T14:32:00Z",

  "source": "youtube|fleurs|indicvoices",
  "source_id": "string-id",
  "source_url": "https://... or null",
  "chunk_index": 0,

  "language": "hi",
  "language_confidence": 0.94,
  "language_method": "mms_lid_2048|whisper_lid|tie_break|inherited_from_neighbor",
  "language_secondary": {"language": "hi", "confidence": 0.91, "method": "whisper_lid"},
  "lid_inherits_from_neighbor": false,

  "transcript": "<source-language string>",
  "translation_en": "<English string or null>",
  "transcript_method": "faster_whisper_large_v3|mms_1b_all",
  "transcript_avg_logprob": -0.32,

  "duration_s": 12.4,
  "sample_rate": 24000,
  "num_channels": 1,
  "num_speakers": 2,
  "speaker_segments": [
    {"speaker": "S0", "start": 0.0, "end": 3.2},
    {"speaker": "S1", "start": 3.2, "end": 7.8}
  ],

  "snr_db": 18.5,
  "speech_ratio": 0.78,
  "clip_ratio": 0.001,
  "music_likely": false,

  "stream_layout": "single|dual",
  "tokenizer_config": {
    "sp_model": "omnivoxtral_sp.model",
    "text_hz": 5,
    "mimi_num_quantizers": 8,
    "stride": 21
  },
  "token_count": 505,
  "token_range": [0, 81920],
  "valid_token_mask": [true, true, ..., false, false],

  "data_license_class": "cc-by-nc|apache|mit",
  "quarantine_reason": null
}
```

### 4.2 Loader contract

`VoxtralDataset` rejects sidecars where:
- `schema_version != 2` (raise, do not fall back).
- `language` is missing or null → tag sample as `"unknown"`, count as missing for the 5% threshold.
- `valid_token_mask` length doesn't match `token_count` → raise.
- `tokenizer_config.stride` doesn't match the architectural expectation for `stream_layout` (21 for single, 42 for dual) → raise.

## 5. Test contract (`tests/`)

### 5.1 `tests/conftest.py` fixtures

- `synthetic_hindi_audio`: 5s Hindi TTS-generated, 24kHz mono, cached locally.
- `synthetic_two_speaker_audio`: 30s concatenation of two TTS speakers with known turn boundaries.
- `silence_5s`: 5s bit-zero, 24kHz mono.
- `noise_5s_minus_45db`: 5s Gaussian noise scaled to -45 dB RMS, 5s.
- `room_tone_5s`: 5s extracted from a quiet region of an existing FLEURS sample.
- `tokenizer_v2`: initialized v2 `VoxtralTokenizer`; module-scoped.

### 5.2 Test files (one per phase + regression)

| File | Coverage |
|---|---|
| `test_phase0_preflight.py` | Phase 0 (SP roundtrip, Mimi silence diff, LID pilot) |
| `test_tokenizer_v2.py` | Phase 1 (encode/translate/sidecar I/O, valid_token_mask, default config) |
| `test_vad_chunker.py` | Phase 2 (mono/stereo regions, backchannel preservation, atomic chunk.json writes) |
| `test_language_detection.py` | Phase 3 (LID agreement, streaming context, MMS routing for 7 langs) |
| `test_dual_stream_pipeline.py` | Phase 4 (speaker segmentation, silence-as-non-zero-Mimi, sidecar schema) |
| `test_train_omni_smoke.py` | Phase 5 (5-step training run, stride homogeneity, mean-across-ranks val) |
| `test_no_gibberish.py` | Regression (Whisper LID confidence, autocorrelation, F0 std) |
| `test_interruption_emission.py` | Goal 4 (synthetic dual-stream, silence emission ≥70%) |

### 5.3 Run command

```bash
uv run pytest tests/                            # full suite
uv run pytest tests/test_no_gibberish.py -v     # regression on each checkpoint
```

### 5.4 CI integration (lightweight)

No external CI infrastructure provisioned. PR template requires:
- "All tests pass on local A10G" checkbox.
- "Phase 0 pre-flight has run successfully" checkbox.
- WER results from `eval_wer.py` for the latest checkpoint pasted into PR description.

## 6. Configuration changes (env vars)

| Env var | Default | Purpose |
|---|---|---|
| `DATA_PATH` | `./data/tokens_v2` | Phase 1a/1b output. Trainer reads this. |
| `DATA_PATH_DUAL` | `./data/tokens_v2_dual` | Phase 4 dual-stream output. Used when `DUAL_STREAM=true`. |
| `DUAL_STREAM` | `false` | Enables stride-42 dual-stream batching. |
| `VAL_SPLIT_PATH` | `./data/val_split_v2.json` | Pinned validation file list. |
| `SP_TOKENIZER_PATH` | `./data/tokenizer/omnivoxtral_sp.model` | SentencePiece model. |
| `WHISPER_PATH` | `openai/whisper-large-v3` | Primary ASR model. |
| `MMS_LID_PATH` | `facebook/mms-lid-2048` | Primary LID model. |
| `MMS_ASR_PATH` | `facebook/mms-1b-all` | ASR fallback for unsupported Indic. |
| `LID_CONFIDENCE_THRESHOLD` | `0.7` | Below this → quarantine. |
| `STREAMING_LID_THRESHOLD_S` | `4.0` | Chunks shorter than this → ±2s neighbor context. |
| `VAD_MIN_SPEECH_MS` | `300` | Silero VAD minimum speech segment. |
| `VAD_MIN_SILENCE_MS` | `500` | Silero VAD minimum silence between segments. |
| `VAD_PAD_MS` | `150` | Silero VAD onset/offset preserve. |
| `VAD_MAX_SPEECH_S` | `20` | Silero VAD upper bound. |
| `TRANSLATION_SAMPLE_RATE` | `0.10` | Fraction of chunks that get an English translation sidecar. |
| `STEREO_CORRELATION_THRESHOLD` | `0.5` | Below → speaker-per-channel; above → diarize. |
| `SILENCE_STRATEGY` | `"noise_floor"` | "noise_floor" or "room_tone" — set by Phase 0 result. |
| `USE_DEMUCS` | `auto` | `auto` reads `data/source_music_flags.json`; `true`/`false` overrides. |
| `SIDECAR_MISSING_THRESHOLD` | `0.05` | Trainer raises if more than this fraction of files lack sidecars. |
| `LANGUAGE_TAG_MODE` | `"per_utterance"` | "per_utterance" or "global_prepend" — set by Phase 0 SP roundtrip result. |

## 7. Dependency additions to `pyproject.toml`

```toml
[project]
dependencies = [
  # existing...
  "silero-vad>=6.0",
  "faster-whisper>=1.0",
  "ctc-forced-aligner>=1.0",
  "pyannote.audio>=3.1",
  "demucs>=4.0",                          # opt-in via USE_DEMUCS=true
  "jiwer>=3.0",                           # for WER computation
  "utmos>=0.1",                           # for naturalness scoring
]
```

## 8. Phase-by-phase implementation checklist (for the implementer)

### Phase 0
- [ ] `scripts/phase0_preflight.py` — 3 experiments + report writer.
- [ ] `tests/test_phase0_preflight.py` — unit coverage for each experiment.
- [ ] Run: `uv run scripts/phase0_preflight.py`. Verify `phase0_results.md` exists.
- [ ] Update `claude-plan.md` and config defaults if any gate is hit.

### Phase 1a
- [ ] Modify `VoxtralTokenizerConfig` defaults (sp_tokenizer_path, language).
- [ ] Modify `VoxtralTokenizer.encode` signature (per-call language, task, source_metadata).
- [ ] Add `VoxtralTokenizer.translate`.
- [ ] Modify `TimedWhisperTokenizer` (per-call language, task, drop conditional `if language != "en"`).
- [ ] Add `write_metadata_sidecar` to `preprocessing.py`.
- [ ] `_save_tokens` becomes atomic (temp + rename).
- [ ] Add `valid_token_mask` to `compute_omni_loss`.
- [ ] `tests/test_tokenizer_v2.py` passes.
- [ ] `scripts/retokenize_v2.py` for FLEURS + IndicVoices.
- [ ] Run: `bash autoresearch/scripts/run_safe.sh "uv run scripts/retokenize_v2.py --dataset fleurs --languages hi,ta,bn,gu,mr,pa,ur,ne,or,as,ml,kn,te" 86400`
- [ ] Verify `data/tokens_v2/fleurs/<lang>/` populated.

### Phase 1.5
- [ ] `VoxtralDataset` reads `valid_token_mask` from sidecar; defaults to all-True.
- [ ] `compute_omni_loss` zeros loss on masked positions.
- [ ] `test_train_omni_smoke.py` covers mask handling.

### Phase 2
- [ ] `scripts/vad_chunker.py`.
- [ ] `tests/test_vad_chunker.py` including backchannel + stereo cases.
- [ ] Run on `data/chunks_indic_yt/` → `data/chunks_v2/yt/`.

### Phase 3
- [ ] `scripts/detect_language.py` with batched-pass architecture.
- [ ] `src/voxtral/tokenizer/mms_asr.py`.
- [ ] `tests/test_language_detection.py`.
- [ ] Run on `data/chunks_v2/yt/` → produces `<chunk>.lang.json` files.

### Phase 1b
- [ ] Run `retokenize_v2.py --dataset youtube --input_path data/chunks_v2/yt`.
- [ ] Verify `data/tokens_v2/yt/<lang>/` populated.

### Phase 4
- [ ] `scripts/detect_music.py`.
- [ ] `scripts/diarize_v2.py`.
- [ ] `DualStreamTokenizer.encode_with_metadata`.
- [ ] `tests/test_dual_stream_pipeline.py`.
- [ ] Run on YouTube source audio → `data/dual_chunks_v2/yt/` → `data/tokens_v2_dual/yt/`.

### Phase 5
- [ ] `VoxtralDataset` upgrades (sidecar reader, temperature sampling, stride homogeneity, pinned val split, `random.Random`).
- [ ] `compute_omni_loss` mean-across-ranks fix.
- [ ] `data/val_split_v2.json` captured at Phase 1a cutover.
- [ ] `scripts/eval_wer.py`.
- [ ] `tests/test_no_gibberish.py`.
- [ ] `tests/test_interruption_emission.py`.
- [ ] Fresh-init training run started on v2 data. Monitor val_loss, WER, and gibberish regression test every 1,000 steps.

## 9. Out-of-scope clarifications

- No model architectural changes. `OmniVoxtral` and `DepthTransformer` stay as-is.
- No changes to `mimi/` (vendored Kyutai code; don't touch).
- No changes to `autoresearch/prepare.py` or `autoresearch/train.py` (upstream Karpathy harness, separate contract).
- Inference-side fixes (rep penalty, RAS, embedding norm) tracked separately; not in this LLD.

## 10. Open implementation questions (to resolve during Phase 0)

1. **SP `<lang_xx>` injection mode.** If the SP roundtrip test passes, use per-utterance prefix. If fails, switch to global prepend (one slot at sequence start). `LANGUAGE_TAG_MODE` env var captures the choice.

2. **Silence encoding strategy.** If Phase 0 experiment 2 shows -45 dB noise differs from bit-zero in Mimi q1 codes, use `SILENCE_STRATEGY="noise_floor"`. If indistinguishable, use `"room_tone"`.

3. **Phase 3 confidence threshold.** Default 0.7, adjusted based on the 200-chunk pilot. If pilot rejection > 50%, lower to 0.6 and require streaming-context for all chunks.
