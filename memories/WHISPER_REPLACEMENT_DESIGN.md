# Whisper Replacement Design: Multilingual Word-Level Timestamps

> Design document for replacing `whisper-tiny.en` with a multilingual alignment system.
> Date: 2026-03-13 | Phase 2, Step 2.3

---

## Problem Statement

Current `TimedWhisperTokenizer` (`src/voxtral/tokenizer/word_level_whisper.py`) uses:
- `whisper-tiny.en` (39M params, English-only)
- Cross-attention weights for word-level timestamps
- Mistral's 32K BPE tokenizer for text encoding (catastrophic 10.8 fertility on Indic)

**Requirements for OmniVoxtral:**
1. Support 22 Indic languages + English
2. Word-level timestamp accuracy ≤ 50ms for interleaving with audio tokens
3. Fit on A10G alongside Mimi (~300MB) within 23GB VRAM
4. Use our new SentencePiece 65K tokenizer for text output
5. Streaming-capable (for future real-time inference)

---

## Options Analysis

### Option A: Whisper-large-v3 (direct replacement)

**How:** Swap `whisper-tiny.en` → `openai/whisper-large-v3` in `TimedWhisperTokenizer`.

| Attribute | Value |
|-----------|-------|
| Model size | 1.55B params (~3GB fp16) |
| Languages | 99+ (all 22 Indic covered) |
| Word timestamps | Yes, via `return_token_timestamps=True` |
| Indic WER | 15-30% (language-dependent, FLEURS benchmarks) |
| Streaming | No native streaming (30s chunks) |
| A10G fit | Yes (~3GB + Mimi 300MB = 3.3GB, leaves 19GB) |

**Pros:**
- Drop-in replacement — minimal code changes to `word_level_whisper.py`
- Well-tested, stable API via `transformers`
- Built-in word timestamp support via cross-attention
- Language detection built-in (`model.detect_language()`)

**Cons:**
- Word timestamp quality varies across languages (English > Hindi > Dravidian)
- No streaming — must process 30s chunks
- Large model (3GB) — bigger than needed if only used for alignment
- Cross-attention timestamps are noisy for long utterances

### Option B: Whisper-large-v3 + WhisperX (forced alignment)

**How:** Use Whisper-large-v3 for transcription, then wav2vec2/MMS model for forced alignment.

| Attribute | Value |
|-----------|-------|
| ASR model | Whisper-large-v3 (1.55B) |
| Alignment model | wav2vec2-BERT or MMS-1B (1B params) |
| Languages | 1100+ (MMS covers all 22 Indic) |
| Word timestamps | Yes, CTC-based forced alignment |
| Accuracy | Typically ±20-30ms (better than cross-attention) |
| A10G fit | Tight — 3GB + 2GB = 5GB for both models |

**Pros:**
- Best word-level timestamp accuracy (CTC alignment is more precise than cross-attention)
- WhisperX is well-maintained open-source project
- MMS alignment models exist for 1100+ languages (all 22 Indic covered)

**Cons:**
- Two models in memory (Whisper + alignment model)
- Two-pass pipeline: slower (ASR → alignment)
- WhisperX dependency adds complexity
- Not streaming — requires full utterance for alignment

### Option C: MMS (Meta Massively Multilingual Speech)

**How:** Use `facebook/mms-1b-all` for CTC-based ASR, extract timestamps from CTC alignment.

| Attribute | Value |
|-----------|-------|
| Model size | 1B params (~2GB fp16) |
| Languages | 1162 languages (all 22 Indic + many dialects) |
| Word timestamps | Via CTC forced alignment (character-level) |
| Indic WER | Often lower than Whisper for low-resource Indic |
| Streaming | CTC models can stream with chunk-based decoding |
| A10G fit | Yes (~2GB) |

**Pros:**
- Best Indic language coverage (1162 languages)
- Smaller than Whisper-large-v3
- CTC alignment gives precise character-level timestamps
- Potentially better WER on low-resource Indic languages (MMS halved Whisper's WER on FLEURS)
- CTC architecture is streaming-friendly

**Cons:**
- No native word-level segmentation (must aggregate character timestamps to words)
- Language-specific adapter switching required per utterance
- Less mature inference tooling than Whisper
- No built-in transcript + timestamp joint output

### Option D: IndicWhisper (AI4Bharat fine-tuned)

**How:** Use AI4Bharat's IndicWhisper models fine-tuned for Indic languages.

| Attribute | Value |
|-----------|-------|
| Base model | Whisper-large-v2 fine-tuned |
| Languages | 12 Indic languages |
| Word timestamps | Via Whisper cross-attention (inherited) |
| Indic WER | Typically 5-15% better than vanilla Whisper on supported languages |
| Coverage | Only 12/22 languages (missing Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Sanskrit, Santali, Sindhi, Odia) |

**Pros:**
- Best WER on supported Indic languages
- Same API as Whisper (drop-in for supported languages)

**Cons:**
- Only 12/22 languages — incomplete coverage
- Based on Whisper-large-v2 (not v3)
- Still has cross-attention timestamp limitations
- Would need fallback to vanilla Whisper or MMS for missing languages

### Option E: Hybrid (MMS language detection + Whisper-large-v3 alignment)

**How:** MMS for language identification, Whisper-large-v3 for ASR + timestamps, with MMS CTC alignment as fallback.

| Attribute | Value |
|-----------|-------|
| LangID model | facebook/mms-lid-4017 (256MB) |
| ASR model | openai/whisper-large-v3 (3GB) |
| Fallback alignment | facebook/mms-1b-all CTC (2GB, loaded on demand) |
| Languages | 22+ Indic fully covered |

**Pros:**
- Robust language detection before ASR
- Best-of-both-worlds: Whisper quality + MMS coverage
- MMS LangID is tiny (256MB) and very accurate

**Cons:**
- Most complex pipeline (3 models)
- Highest memory usage if all loaded simultaneously
- More code to maintain

---

## UPDATED Recommendation: Two-Stage Pipeline (ASR + CTC Forced Alignment)

> Updated based on deep research (2024-2025 papers). Original Option A recommendation revised.

### Critical Finding: Whisper's Brahmic Text Normalization Bug

OpenAI's reported Whisper WERs for Indic languages are **artificially deflated** by a text
normalization bug that collapses Brahmic characters. True error inflation:
- Hindi: +21.9% higher WER than reported
- Tamil: +41.5% higher WER than reported
- Malayalam: +152.2% higher WER than reported

Source: [Deepgram analysis](https://deepgram.com/learn/how-openai-s-text-normalization-hides-whisper-s-true-word-error-rate-for-south-asian-and-southeast-asian-languages)

This means vanilla Whisper-large-v3 has **true Hindi WER of 25-35%**, not the reported 15-20%.

### New Option F: IndicWhisper ASR + MMS_FA CTC Alignment (RECOMMENDED)

**How:** Two-stage pipeline decoupling ASR quality from alignment quality.

**Stage 1 — ASR:** IndicWhisper or IndicConformer (AI4Bharat fine-tuned) for transcript.
**Stage 2 — Alignment:** `torchaudio.pipelines.MMS_FA` (300M) for word-level timestamps via CTC forced alignment.

| Attribute | Value |
|-----------|-------|
| ASR model | IndicConformer-600M or IndicWhisper-large (1.55B) |
| Alignment model | MMS_FA (300M, torchaudio native) |
| Languages | 22 Indic (ASR) + 1,130 (alignment) |
| Word timestamps | CTC Viterbi alignment (character → word level) |
| Accuracy | ±20-30ms (better than Whisper cross-attention) |
| Alignment speed | <100ms per 30s segment on GPU |
| A10G fit | 1.55B + 300M = ~3.7GB, fits easily |

**Why this is better than Option A (Whisper-large-v3 alone):**
1. CTC alignment is architecturally better for frame-level timing than autoregressive attention
2. MMS_FA is purpose-built for alignment on 23K hours across 1,130 languages
3. Decouples ASR from alignment — bad ASR doesn't corrupt timestamps if transcript is corrected
4. IndicWhisper has 4.1% lower WER on Indic than vanilla Whisper (39/59 Vistaar benchmarks)
5. MMS_FA alignment runs 10-50x real-time (vs Whisper cross-attention which adds 0 cost but lower quality)

### Phased Implementation

**Phase 2 (now):** Replace `whisper-tiny.en` with `openai/whisper-large-v3` as the quick win.
This is a ~10 line change and unblocks multilingual training data prep immediately.

**Phase 3 (data pipeline):** Add MMS_FA CTC forced alignment as post-processing.
Uses `torchaudio.functional.forced_align()` — native PyTorch, no extra dependencies.
Replace Whisper with IndicWhisper/IndicConformer for ASR on Indic languages.

**Phase 4 (streaming):** IndicConformer's RNNT variant supports streaming ASR.
CTC emissions can be computed in streaming chunks for quasi-streaming alignment.

### Recent Papers (2024-2025)

- **"Whisper Has an Internal Word Aligner"** (Yeh et al., Sep 2025, arXiv:2509.09987) — Selecting specific attention heads + character-level teacher forcing yields 33-40% alignment improvement
- **"CrisperWhisper"** (INTERSPEECH 2024, arXiv:2408.16589) — Improved word timestamps via tokenizer modification + DTW (English-focused)
- **"Enhancing Whisper for Indian Languages"** (Dec 2024, arXiv:2412.19785) — Prompt-tuning with language family info for 8 Indic languages
- **"Real-Time Word-Level Temporal Segmentation"** (Apr 2025, arXiv:2504.10849) — Methods for streaming word timestamps
- **IndicVoices-R** (NeurIPS 2024) — 1,704 hours across all 22 Indic languages

---

## Implementation Plan

### Changes to `word_level_whisper.py`:

```python
# BEFORE (line 176-189):
class TimedWhisperTokenizer(torch.nn.Module):
    def __init__(self, model_name: str, hertz: int) -> None:
        ...
        self.language: str = "en"
        self.mistral_tokenizer = tr.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3", ...)

# AFTER:
class TimedWhisperTokenizer(torch.nn.Module):
    def __init__(self, model_name: str, hertz: int, language: str = "en",
                 text_tokenizer_path: str | None = None) -> None:
        ...
        self.language: str = language
        if text_tokenizer_path and text_tokenizer_path.endswith(".model"):
            import sentencepiece as spm
            self.sp_tokenizer = spm.SentencePieceProcessor()
            self.sp_tokenizer.load(text_tokenizer_path)
            self.use_sp = True
        else:
            self.mistral_tokenizer = tr.AutoTokenizer.from_pretrained(...)
            self.use_sp = False
```

### Changes to `tokenizer/model.py`:

```python
# BEFORE (line 16):
model_name: str = "openai/whisper-tiny.en"

# AFTER:
model_name: str = "openai/whisper-large-v3"
```

### Changes to `generate_tokens()`:

```python
# BEFORE:
def generate_tokens(processor, model, audio):
    ...
    return model.generate(input_features, return_timestamps=True,
                         return_token_timestamps=True)

# AFTER:
def generate_tokens(processor, model, audio, language="en"):
    ...
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    return model.generate(input_features, return_timestamps=True,
                         return_token_timestamps=True,
                         forced_decoder_ids=forced_decoder_ids)
```

### New: Language detection helper

```python
def detect_language(processor, model, audio):
    """Detect language from audio using Whisper's built-in detection."""
    input_features = processor(
        audio.numpy(), sampling_rate=16000, return_tensors="pt"
    ).input_features.to(next(model.parameters()).device)

    lang_ids = model.detect_language(input_features)
    return lang_ids  # Returns predicted language code
```

---

### Phase 3 addition: MMS_FA CTC Forced Alignment

```python
import torchaudio

# Load MMS forced alignment pipeline
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model()  # 300M params
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

def align_transcript(audio_waveform, transcript, language="hin"):
    """Get word-level timestamps via CTC forced alignment."""
    # Get CTC emissions
    with torch.no_grad():
        emission, _ = model(audio_waveform)

    # Tokenize transcript
    tokens = tokenizer(transcript)

    # Force align
    token_spans = aligner(emission[0], tokens)

    # Aggregate character spans to word spans
    word_spans = []
    words = transcript.split()
    char_idx = 0
    for word in words:
        word_start = token_spans[char_idx].start
        word_end = token_spans[char_idx + len(word) - 1].end
        word_spans.append((word, word_start, word_end))
        char_idx += len(word) + 1  # +1 for space

    return word_spans
```

---

## Memory Budget (A10G 23GB)

### Phase 2 (Whisper-only, immediate):
| Component | Size | Notes |
|-----------|------|-------|
| Whisper-large-v3 | 3.0 GB | fp16, or Turbo 809M = 1.6GB |
| Mimi codec | 0.3 GB | 8 quantizers |
| SentencePiece tokenizer | <1 MB | CPU-only |
| Audio buffers + overhead | 1.0 GB | |
| **Total** | **4.3 GB** | Leaves 18.7GB for training |

### Phase 3 (Two-stage pipeline, future):
| Component | Size | Notes |
|-----------|------|-------|
| IndicWhisper/IndicConformer | 1.2-3.0 GB | ASR only |
| MMS_FA (300M) | 0.6 GB | CTC forced aligner |
| Mimi codec | 0.3 GB | 8 quantizers |
| SentencePiece tokenizer | <1 MB | CPU-only |
| Audio buffers + overhead | 1.0 GB | |
| **Total** | **3.1-4.9 GB** | Leaves 18-20GB for training |

---

## Whisper Language Codes for 22 Indic Languages

| Language | ISO 639-3 | Whisper code | FLEURS code | Status |
|----------|-----------|-------------|-------------|--------|
| Assamese | asm | as | as_in | Supported |
| Bengali | ben | bn | bn_in | Supported |
| Bodo | brx | — | — | NOT supported (use Hindi fallback) |
| Dogri | doi | — | — | NOT supported (use Hindi fallback) |
| Gujarati | guj | gu | gu_in | Supported |
| Hindi | hin | hi | hi_in | Supported |
| Kannada | kan | kn | kn_in | Supported |
| Kashmiri | kas | — | — | NOT supported (use Urdu fallback) |
| Konkani | kok | — | — | NOT supported (use Marathi fallback) |
| Maithili | mai | — | — | NOT supported (use Hindi fallback) |
| Malayalam | mal | ml | ml_in | Supported |
| Manipuri | mni | — | — | NOT supported (use Bengali fallback) |
| Marathi | mar | mr | mr_in | Supported |
| Nepali | nep | ne | ne_np | Supported |
| Odia | ori | — | or_in | NOT in Whisper (use MMS fallback) |
| Punjabi | pan | pa | pa_in | Supported |
| Sanskrit | san | sa | — | Supported (limited quality) |
| Santali | sat | — | — | NOT supported (use MMS) |
| Sindhi | snd | sd | — | Supported |
| Tamil | tam | ta | ta_in | Supported |
| Telugu | tel | te | te_in | Supported |
| Urdu | urd | ur | ur_pk | Supported |

**Coverage: 15/22 languages natively. 7 languages need fallback (MMS or related-language Whisper).**

For the 7 unsupported languages (Bodo, Dogri, Kashmiri, Konkani, Maithili, Manipuri, Santali), the fallback strategy is:
1. Use MMS-1B for ASR (supports all 22)
2. Use related-language Whisper for timestamp quality comparison
3. In Phase 3, train language-specific adapters for these 7

---

## Verification

After implementing the Whisper replacement:

```bash
# Test that timestamps work for each supported language
CUDA_VISIBLE_DEVICES=0 uv run python -c "
from voxtral.tokenizer.word_level_whisper import TimedWhisperTokenizer
import torchaudio

tok = TimedWhisperTokenizer('openai/whisper-large-v3', hertz=5)
tok.language = 'hi'  # Hindi
audio, sr = torchaudio.load('data/indic_test_samples/hin/sample_0.wav')
audio = torchaudio.functional.resample(audio, sr, 16000)
tokens = tok(audio, 16000)
print(f'Token shape: {tokens.shape}')  # Should be [1, num_timesteps * hertz]
"
```

**Pass criteria:**
- Token tensor is non-empty for all 15 Whisper-supported languages
- Timestamps are monotonically increasing
- Text output is recognizable (not garbage) for manual spot-check
