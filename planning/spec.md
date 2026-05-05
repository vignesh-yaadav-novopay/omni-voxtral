# OmniVoxtral Training Pipeline — Ground-Up Rebuild

## Problem statement

After 528 logged experiments, the model reaches val_loss=1.489 (text_acc=97%, depth_acc=66%) on teacher-forced one-step prediction, but **autoregressive generation produces repetitive Hindi-shaped gibberish** ("आपाशाशा…"). Whisper auto-detects the model's own output as Swedish. The architecture is sound; the training data labels are corrupt. A first-principles audit of the data pipeline (`src/voxtral/data/preprocessing.py`, `src/voxtral/tokenizer/model.py`, `src/voxtral/tokenizer/word_level_whisper.py`, `src/voxtral/tokenizer/dual_stream.py`, `scripts/preprocess_hf.py`, `scripts/diarize_audio.py`) found five compounding defects that silently destroyed the text stream and erased the prosody/interruption signal that Mimi was supposed to preserve.

This spec defines a ground-up rebuild of the training data pipeline. The goal is a model that:

- Learns prosody (preserves it from Mimi q2-q8 acoustic residuals through to generation).
- Learns when to speak vs listen (interruptions, backchannels, turn-taking) from real conversational data.
- Produces intelligible speech in 22 Indic languages.

## The 5 defects (from systematic investigation)

### Defect 1 — Whisper language is hardcoded at construction time, not detected per file

`VoxtralTokenizerConfig.language` is set once when the tokenizer is built and reused for every file. `preprocess_hf.py:175` instantiates with the default `language="en"`. The Whisper code path at `word_level_whisper.py:177-180` only forwards `language` and `task` to `model.generate()` when `language != "en"` — so all 57,668 YouTube chunks and all FLEURS/IndicVoices files were processed with no explicit language hint. Whisper falls back to its own LID, which is unreliable on:

- Short chunks under 4 seconds (LID needs context).
- Code-mixed Indic-English podcasts (most YouTube content).
- Conversational IndicVoices where two speakers interleave.

The detected language is **never logged or persisted** — there is no way to audit what Whisper decided for each file, and no mechanism to reject low-confidence detections.

### Defect 2 — Whisper transcripts are re-tokenized with Mistral BPE for Indic scripts

`word_level_whisper.py:206-213` falls back to Mistral BPE when `sp_tokenizer_path` is None — which is the default. Mistral BPE has fertility ~10.79 tokens/word for Devanagari. Combined with hard truncation to `text_hz=5` tokens per 1-second bucket (`word_level_whisper.py:226-228`), most Indic transcripts get sliced to the first 5 byte-level pieces of the first word. The remainder is silently dropped.

A SentencePiece Unigram 65K tokenizer with fertility 1.91 for Indic was trained and lives at `data/tokenizer/omnivoxtral_sp.model`, but no preprocessing run has ever used it because the default config is `sp_tokenizer_path=None`.

### Defect 3 — `task="translate"` is never an option in the code

`generate_tokens()` at `word_level_whisper.py:163-180` hardcodes `task="transcribe"` whenever it sets a task at all. There is no path to extract Whisper's English translation. This eliminates a useful auxiliary signal for evaluation and cross-lingual grounding.

### Defect 4 — Mimi preserves prosody, but the pipeline throws prosody away

The Mimi codec is the only correctly-functioning component: q1 carries WavLM-distilled semantics, q2-q8 carry acoustic residuals encoding speaker timbre, F0 contour, micro-timing, environment. The pipeline destroys this signal four ways:

1. **20-second hard chunking** (`preprocessing.py:27`, `preprocess_hf.py:58`). YouTube turn lengths are 30-90s; chunking at 20s splits utterances mid-sentence and severs prosody contours.
2. **Pad-to-fixed-length with zeros** (`preprocessing.py:81-84`). Files shorter than 20s are zero-padded. Mimi encodes silence as a stable repeated token pattern. The model learns "emit this silent loop forever" — exactly the inference repetition mode.
3. **Stereo → mono mean** (`preprocess_hf.py:124-125`, `diarize_audio.py:58-59`). Many podcast YouTube uploads have host/guest on separate channels. Averaging destroys channel separability that diarization could exploit.
4. **Single-stream tokenization on conversational corpora.** All 57,668 YouTube chunks were tokenized through `preprocess_hf.py` (single-stream), not through `diarize_audio.py` (dual-stream). The model never saw an interruption or turn-taking event in training. There is no `data/dual_tokens/` directory on disk — diarization was never run on the YouTube corpus.

### Defect 5 — No metadata sidecars persist alongside tokens

`preprocess_hf.py:241` saves `tokens.numpy()` and nothing else. No language tag, no Whisper confidence, no transcript, no translation, no speaker count, no duration, no source URL. Consequences:

- Cannot apply per-language temperature sampling (τ=3.3 for Indic balancing).
- Cannot audit what Whisper actually decided per file.
- Cannot mask windows where Whisper confidence was low.
- Cannot retokenize text without re-encoding all the audio (the expensive step).

## Required behavior of the rebuilt pipeline

### Per-chunk language handling

- Run Whisper LID **once per chunk** and persist the result with confidence score.
- For chunks where Whisper LID confidence is below a threshold (suggested 0.8) or where the language is one of the 7 Indic languages Whisper-large-v3 doesn't support (Bodo, Dogri, Konkani, Maithili, Manipuri, Santali, Sindhi), fall back to MMS LID or skip the file.
- Language tag is **not** baked into a tokenizer instance — it is passed as a per-call argument so the same Whisper model can serve every language.
- For unreliable short chunks (under ~4 seconds), use a streaming-Whisper path inspired by the realtime references (`https://github.com/antirez/voxtral.c`, `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602`). Streaming windows accumulate context before LID is committed.
- Optional integration with `https://github.com/MahmoudAshraf97/whisper-diarization` for joint timed-Whisper + speaker diarization in a single pass (avoids running Whisper and pyannote separately).

### Speaker diarization for conversational data

- All conversational sources (YouTube podcasts, IndicVoices conversational subset) go through diarization **before** tokenization, not as a separate optional step.
- Diarization output drives dual-stream tokenization: the speaker with more total speech is the "model" stream, the other is the "user" stream (or both streams in symmetric duplex mode).
- Stereo source audio is preserved through the pipeline — left/right channel separation is exploited as an additional diarization signal where available, instead of being averaged away in step 1.
- Silence between turns is encoded as a **learnable signal**, not as zero-padded filler. The model needs to learn "this is when I should listen, not speak."

### Tokenizer fixes

- Default `VoxtralTokenizerConfig.sp_tokenizer_path` becomes `data/tokenizer/omnivoxtral_sp.model` for all Indic-touching runs. Mistral BPE remains available as an opt-in for English-only experiments.
- `TimedWhisperTokenizer` exposes `task` as a per-call argument so callers can request `"transcribe"` (source-language transcript) or `"translate"` (English translation).

### Inner monologue + translation sidecar

- Source-language transcript drives the inner monologue text stream (text-leads-audio causality preserved). This is **Option A** from the prior investigation.
- An English translation is generated by Whisper in a second pass and stored as **sidecar metadata only** — not interleaved into the training tokens. It exists for evaluation (BLEU on transcribed-then-translated output, cross-lingual grounding tests) and for downstream English-prompted Indic TTS experiments without retraining the main loss.

### Variable-length chunking with VAD

- Replace 20-second hard chunking with VAD-aware variable-length chunking. Cut at silence boundaries (~0.5s of low energy) up to a soft cap (~30 seconds).
- No zero-padding. The dataset uses bucketed sequence lengths; the data loader already supports packing (`src/voxtral/trainer/data.py`).
- Pre-roll and post-roll context: keep ~200ms before and after each utterance to preserve onsets and decays.

### Metadata sidecars

For every `<file>.npy`, write a paired `<file>.meta.json` containing:

```json
{
  "language": "hi",
  "language_confidence": 0.94,
  "transcript": "आज मौसम बहुत अच्छा है",
  "translation_en": "The weather is very good today",
  "duration_s": 12.4,
  "num_speakers": 2,
  "speaker_segments": [
    {"speaker": "S0", "start": 0.0, "end": 3.2},
    {"speaker": "S1", "start": 3.2, "end": 7.8}
  ],
  "source": "youtube",
  "source_id": "<video-id>",
  "chunk_index": 0,
  "preprocessing_version": "v2",
  "whisper_model": "openai/whisper-large-v3",
  "diarization_model": "pyannote/speaker-diarization-3.1",
  "sp_tokenizer": "omnivoxtral_sp.model"
}
```

The data loader can then enforce language sampling, mask low-confidence windows, and re-tokenize text without touching the audio.

## YouTube corpus is the primary target

- 57,668 chunks already on disk at `data/chunks_indic_yt/`.
- 1,845 source URLs in `data/indic_urls.txt` (~2,654 hours of indexed Indic content).
- Existing token files `data/tokens_yt/` and `data/tokens_yt_g0..g3/` were preprocessed with the corrupted pipeline and **must be discarded or quarantined**, not migrated.
- Re-tokenization budget is ~24-36 GPU-hours for diarization on a single A10G plus ~200 GPU-hours for Whisper+Mimi. Plan must phase this so training is not blocked for that window.

## Hardware constraints

- Dev machine: 1× A10G (24 GB).
- Production training: 4× A10G DDP (current setup).
- Target deployment: 2× H100.
- All preprocessing must fit on a single A10G with the watchdog (`autoresearch/scripts/run_safe.sh`) supervising RAM/GPU thresholds — there is documented history of RAM pegs causing SSH lockout.

## Decision required in the plan

**Option A — Source-language inner monologue (recommended).** Text tokens are in the same language as the audio. Whisper detects language, transcribes in source. Inner Monologue's "the model thinks in the language it speaks" is preserved. Translation is sidecar-only.

**Option B — English-driven cross-lingual TTS.** Text tokens are English (Whisper translation), audio tokens are Indic. Good for English-prompted Indic TTS, breaks Inner Monologue's causal alignment trick.

The plan assumes Option A unless specified otherwise. Option B is a straightforward variant once Option A is built.

## Goals and non-goals

### Goals

- Eliminate gibberish at inference. Concrete metric: Whisper-large-v3 WER on generated Hindi audio drops below 50% (currently >80%).
- Preserve prosody. Concrete metric: F0 correlation between generated and reference > 0.85, UTMOS > 2.5.
- Learn interruptions. Concrete metric: model emits silence tokens during user-stream speech in dual-stream eval >70% of the time.
- Make the pipeline auditable. Every token file has a paired metadata sidecar.
- Phase delivery so the training loop can resume on the cleanest available data within ~1 week, full corpus within ~1 month.

### Non-goals

- Architectural changes to the model itself (no MPD, no FSQ codec swap, no language adapters in this rewrite — all gated behind Phase 6 follow-up).
- Inference-side fixes (repetition penalty, RAS sampling, embedding norm rescaling) — these are tracked separately in `experiment_queue.md` as Phase NOW; this rewrite is Phase NEXT (training-side).
- 22-language coverage in v2.0 — start with 13 languages we already have FLEURS data for, expand as IndicVoices-R is processed.

## Phasing constraint

The user has explicitly asked for phased implementation. The plan must define at least:

- **Phase 1 — Tokenizer fixes only** (defects 2, 3, 5). Smallest blast radius. Can re-tokenize the existing chunked audio without re-chunking. Unblocks training within days.
- **Phase 2 — VAD-aware re-chunking + metadata** (defect 4 partial). Replaces the 20s hard cut. New chunks under `data/chunks_v2/`.
- **Phase 3 — Per-chunk LID + streaming fallback** (defect 1). Whisper is called with explicit language; sidecar persists what was decided.
- **Phase 4 — Diarization + dual-stream tokenization on YouTube corpus** (defect 4 full). Enables interruption learning. Largest GPU budget.
- **Phase 5 — Wire everything into `train_omni.py` with dual-stream toggle and per-language sampling.**
- **Phase 6 — Architectural follow-ups** (deferred): MPD, embedding rescaling, semantic distillation, FSQ codec exploration.

## Reference repositories and prior art

- `https://github.com/MahmoudAshraf97/whisper-diarization` — joint timed-Whisper + pyannote speaker labels. Avoids running two passes.
- `https://github.com/antirez/voxtral.c` — minimal C implementation of Voxtral-style streaming inference. Reference for streaming-Whisper LID.
- `https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602` — Mistral's realtime variant. Reference for streaming-mode language detection on short inputs.
- Moshi (arXiv:2410.00037) — dual-stream training, Inner Monologue.
- DiSTAR / MaskGCT / SoundStorm — non-AR codebook prediction (Phase 6 follow-up only).
- IndicVoices-R (1704h, 22 languages) — second-stage data scaling target after YouTube pipeline is proven.

## Output format

The user wants three planning documents in `planning/`:

- `plan.md` — narrative plan (the deep-plan workflow's `claude-plan.md`, renamed at the end).
- `HLD.md` — high-level architectural design (data flow diagrams, component boundaries, phase boundaries).
- `LLD.md` — low-level design (file-by-file changes, function signatures, data schemas, dependencies, runtime contracts).

These three files are the deliverables. The deep-plan workflow's intermediate files (`claude-research.md`, `claude-interview.md`, `claude-spec.md`, `claude-plan-tdd.md`, `sections/`) are scaffolding and may remain in `planning/` for traceability but are not the headline output.
