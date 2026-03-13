# Voxtral Codebase Audit

> Ruthlessly honest assessment of existing Voxtral code for OmniVoxtral planning.
> Date: 2026-03-13 | Auditor: Claude | Lines audited: ~3,400

---

## Executive Summary

Voxtral is a clever hackathon MVP with one genuinely novel idea (discrete multimodal tokenization via vocab extension) and solid engineering in the data pipeline. However, it has critical gaps for production: no loss weighting despite config, English-only, no duplex, no streaming, no tests. **Approximately 40% reusable as-is, 30% needs modification, 30% must be rewritten.**

---

## Data Flow 1: AUDIO → TOKENS (`tokenizer/model.py`)

**What it does:** Takes raw audio → produces a flat interleaved token sequence.

- Mimi codec (vendored from Kyutai) encodes audio at 12.5Hz per codebook × 4 quantizers = 50 audio tokens/sec
- TimedWhisper (`word_level_whisper.py`) produces text tokens at 5Hz using Whisper cross-attention weights for word timestamps
- Each quantizer's tokens are offset to avoid collision: `token = mimi_token + q_idx * mimi_vocab_size + text_vocab_size`
- Audio quantizers interleaved: `[q1_t1, q2_t1, q3_t1, q4_t1, q1_t2, ...]`
- Text and audio interleaved with factor `[1, 40]` (1 text token per 40 audio tokens per window)
- 2-window delay: audio shifted 2 windows later to ensure text appears before its corresponding audio

**Strengths:**
- The interleaving math is clean (`model.py:22-41`). `interleave()` and `uninterleave()` are correct, symmetric, and handle arbitrary factor lists.
- Token offset trick for quantizer disambiguation (`model.py:76-86`) is the right approach — avoids collision without any special token overhead.
- Config is a clean `NamedTuple` with sensible defaults (`model.py:14-19`).

**Weaknesses:**
- `whisper-tiny.en` hardcoded (`model.py:16`) — English only, poor alignment quality from tiny model.
- `sample_rate == 24_000` asserted but never configurable (`model.py:69`).
- 2-window delay (`model.py:92-103`) is a hack around imprecise Whisper alignment. Better: use forced alignment (Montreal Forced Aligner) or Whisper-large-v3 with better timestamp quality.
- 4 quantizers limits audio quality. Should benchmark 4 vs 8 before committing.
- No validation of token ranges — corrupt audio could produce out-of-range tokens silently.
- Whisper timestamps are batch-level (one forward pass at `word_level_whisper.py:164-172`), won't work for streaming.

**For OmniVoxtral:** Architecture must change to dual-stream. The interleave/uninterleave math is reusable but the single-stream assumption is fundamental to this file.

---

## Data Flow 2: TOKENS → MODEL (`trainer/trainer.py`)

**What it does:** Feeds interleaved token sequences to Mistral 7B for autoregressive training.

- Model: `transformers.MistralForCausalLM` (not custom — uses HF model definition, `trainer.py:22`)
- Vocab extended: `model.resize_token_embeddings(2**16)` — adds 33,768 new tokens (`trainer.py:46`)
- Layer pruning: drop every Nth layer (`prune_layers=2` drops half, `trainer.py:49-55`). Crude but effective.
- Optional LoRA: all Linear layers targeted, alpha = 2 × rank (`trainer.py:58-72`)
- Loss: `cross_entropy(logits.view(-1, V), targets.view(-1))` — **UNIFORM** (`trainer.py:166-175`)

### CRITICAL BUG: loss_weights not wired

```python
# config.py line 38:
loss_weights: list[int] = [100, 10, 1]  # text, semantic, acoustic

# trainer.py lines 166-175 (compute_loss):
def compute_loss(voxtral, x):
    input_ids = x[:, :-1].contiguous()
    target_ids = x[:, 1:].contiguous()
    outputs = voxtral(input_ids=input_ids)
    logits = outputs.logits
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), target_ids.view(-1)
    )
    # NEVER references config.loss_weights. Uniform CE over all tokens.
```

**Impact:** Text tokens (most informative, carry semantic meaning) are weighted equally with acoustic tokens (least informative, carry fine audio detail). The Moshi paper found that weighting semantic tokens 100x over acoustic tokens was critical for quality. This bug likely explains why the model struggles to produce coherent output.

### OTHER BUGS

1. **`data_step` resume bug** (`trainer.py:144`): On checkpoint resume, `state.train_dataset.data_step = checkpoint["data_step"]` but `data_step` is computed as `state.step * config.batch_size * config.world_size` at save time (`trainer.py:158`). If batch_size or world_size change between runs, resume replays wrong data.

2. **`save_state` data_step mismatch** (`trainer.py:158`): `"data_step": state.step * config.batch_size * config.world_size` doesn't account for the actual `data_step` in the dataset iterator, which increments by `stride` (not batch_size x world_size).

3. **Gradient checkpointing inactive** (`trainer.py:283-284`): Sets `model.gradient_checkpointing = True` as a bare attribute, but HuggingFace models need `model.gradient_checkpointing_enable()` to actually activate it.

4. **StopIteration silently drops batches** (`trainer.py:316-318`): `except StopIteration: continue` means if the dataset exhausts, remaining steps produce nothing — no warning, no re-shuffle.

5. **Public model push** (`trainer.py:344-348`): `push_to_hub(..., private=False)` pushes model weights publicly with no confirmation. Should default to private.

**Training features (good):**
- EMA with Karras schedule (not simple exponential) — `utils.py:140-176`, sophisticated decay
- Linear warmup → linear decay schedule (`trainer.py:111-115`)
- Gradient norm clipping (`trainer.py:188-190`)
- `torch.compile` support (`trainer.py:294-296`)
- DDP with NCCL (`trainer.py:268-273`, `trainer.py:287-288`)
- W&B logging with audio sample generation during training (`trainer.py:217-228`)

**Assumptions that break at scale:**
1. Single-stream: model sees one token sequence. No concept of "my speech" vs "their speech"
2. No concept of turns, silence, or interruption in the token space
3. No language conditioning — cannot switch languages
4. All data loaded from local `.npy` files — no streaming from object storage
5. Hardcoded to bfloat16 (`trainer.py:75`), no mixed-precision training config

---

## Data Flow 3: MODEL → OUTPUT (`server.py`, `trainer/test.py`)

**What it does:** Takes model output tokens → decodes to audio waveform.

### server.py (99 lines)

- Gradio UI: record mic → tokenize → `model.generate()` → detokenize → play
- `ServerConfig` uses pydantic-settings (good pattern, `server.py:9-14`)

**Bugs:**
1. **Hardcoded `output.wav`** (`server.py:55`): Concurrent requests overwrite each other. Race condition.
2. **No streaming**: User waits for entire generation before hearing anything.
3. **No KV cache management**: No persistent sessions, no memory of prior context.
4. **No WebSocket/WebRTC**: Request-response only.
5. **`tokens.unsqueeze(0)`** (`server.py:42`): `encode()` already returns batch dim. Double-batching causes shape error on inference.
6. **No device placement**: Model loaded without specifying device, tokenizer on default device.

### trainer/test.py (109 lines)

- During training: crop input to first half, generate second half, decode, log to W&B
- Adds a 1kHz beep at the midpoint of generated audio (`test.py:87-99`) — clever debugging trick to hear where generation starts
- `@utils.general_exception_handler` decorator (`test.py:17`) — test failures don't crash training (good)

**Issues:**
1. `test()` moves EMA model to GPU, runs generation, moves back to CPU (`test.py:25`, `test.py:61`). During this time, both main model AND EMA are on GPU — could OOM on A10G.
2. No multilingual evaluation — English-only metrics.
3. `decoded_generation.unbind()` at `test.py:57` — if decode returns wrong shape, silent failure under `@general_exception_handler`.

---

## Data Flow 4: DATA PIPELINE (`data/*.py`)

### indexing.py (213 lines) — YouTube search → URL list

- Uses `youtubesearchpython` (unofficial API, `indexing.py:9`)
- Filters by `min_duration=30min` (podcast-length, `indexing.py:15`)
- Multithreaded with configurable workers (`indexing.py:19`)
- Deduplicates URLs (`indexing.py:65-79`)
- Rate limiting with configurable sleep (`indexing.py:170-172`)

**Good:** Robust multithreaded search with timeout handling.
**Bad:** Search terms are English-only (reads from `data/searches.txt`).
**For Indic:** Infrastructure works perfectly. Just needs Indic search terms in native scripts.

### scraping.py (195 lines) — Download + chunk

- `yt-dlp` download + `ffmpeg segment` with `-c copy` (no transcode, `scraping.py:68-83`)
- Hash-based filenames with 2-char subdirectories (`scraping.py:21-24`, `scraping.py:63-65`)
- ThreadPoolExecutor with 64 workers (`scraping.py:17`)

**Genuinely clever engineering. Keep as-is.**
- No-transcode ffmpeg segmentation achieves ~8000x realtime
- Hash-based naming prevents filesystem bottlenecks

**Risks:**
1. `yt-dlp` can break when YouTube changes API. No fallback/retry after version updates.
2. Temp files use UUID (`scraping.py:37`) but cleanup only happens on success (`scraping.py:103`). Crash = orphaned temp files.
3. `os.listdir()` glob for temp file (`scraping.py:52-54`) is fragile — race condition with concurrent downloads.

### preprocessing.py (139 lines) — Batch GPU tokenization → .npy

- PyTorch `Dataset` + `DataLoader` with workers (`preprocessing.py:31-101`)
- Audio resampled to 24kHz, mono, padded/cut to `chunk_frames` (20s, `preprocessing.py:27`)
- `ThreadPoolExecutor` for parallel `.npy` saving (`preprocessing.py:125-131`)
- `torch.compile` support for tokenizer (`preprocessing.py:122-123`)

**Good:** Batch GPU processing is the right pattern. Compile support is a nice touch.

**Bad:**
1. No validation of output `.npy` files — corrupt writes go undetected.
2. No distributed preprocessing (single machine only, acknowledged by TODO at line 1).
3. No language detection — assumes all audio is English.
4. `torchaudio.load()` loads full audio into memory (`preprocessing.py:61`). For most 20s chunks this is fine, but no guard against pathologically long files.
5. `save_executor.shutdown(wait=True)` at end (`preprocessing.py:133`) but no error handling for failed saves.

---

## Security Issues

1. **`weights_only=False`** (`utils.py:101`): `torch.load(..., weights_only=False)` enables arbitrary code execution via unsafe deserialization. Any malicious checkpoint file can execute arbitrary code on load. Must switch to `weights_only=True`.

2. **Public model push** (`trainer.py:344-348`): Pushes to HuggingFace with `private=False`. Could inadvertently publish unreleased model weights.

3. **No input validation on server** (`server.py`): Audio input from Gradio is passed directly to tokenizer with no size limits, format validation, or sanitization.

4. **Broken HF URL parsing** (`utils.py:96-98`): `repo_id = "".join(split_path[1:3])` joins without separator. URL `huggingface.co/org/repo/file.pt` produces `repo_id="orgrepo"` instead of `"org/repo"`.

---

## Module Salvageability Matrix

| Module | Lines | Verdict | Reuse % | Key Issue |
|--------|-------|---------|---------|-----------|
| `tokenizer/mimi/` | ~1,500 | **KEEP** | 95% | Vendored codec works. May need quantizer flexibility (4→8). |
| `tokenizer/model.py` | 147 | **REWRITE** | 30% | interleave/uninterleave math reusable, but single-stream→dual-stream is fundamental change. |
| `tokenizer/word_level_whisper.py` | 217 | **REWRITE** | 10% | English-only (`whisper-tiny.en`). Core approach sound, but implementation too coupled. |
| `trainer/config.py` | 83 | **KEEP+EXTEND** | 90% | pydantic-settings with env priority is excellent pattern. Extend for new params. |
| `trainer/trainer.py` | 360 | **MODIFY** | 60% | Training loop solid. Wire loss_weights, add FSDP, add language conditioning. |
| `trainer/data.py` | 91 | **MODIFY** | 50% | IterableDataset good. Need multilingual loaders, WebDataset support. |
| `trainer/utils.py` | 199 | **KEEP** | 90% | DDP helpers, EMA (Karras), checkpointing all reusable. Fix `weights_only` and HF URL parsing. |
| `trainer/test.py` | 109 | **MODIFY** | 70% | Generation test pattern good. Needs multilingual evaluation. |
| `data/indexing.py` | 213 | **KEEP+EXTEND** | 85% | Works for YouTube. Add Indic search terms. |
| `data/scraping.py` | 195 | **KEEP** | 95% | No-transcode ffmpeg is elegant. Keep as-is. |
| `data/preprocessing.py` | 139 | **MODIFY** | 60% | Add distributed, language detection, validation. |
| `server.py` | 99 | **REWRITE** | 5% | Gradio → async WebSocket. Almost nothing reusable. |
| `scripts/*.py` | ~60 | **KEEP** | 90% | Config + function call pattern works. |

**Total: ~3,400 lines. ~60% reusable/modifiable, ~40% must be rewritten.**

---

## Critical Issues Summary (Priority Order)

| # | Severity | File:Line | Issue |
|---|----------|-----------|-------|
| 1 | **CRITICAL** | `trainer.py:166-175` | `loss_weights=[100,10,1]` never applied in `compute_loss()` |
| 2 | **CRITICAL** | `server.py:55` | Hardcoded `output.wav` — race condition on concurrent requests |
| 3 | **CRITICAL** | `utils.py:101` | `weights_only=False` enables arbitrary code execution |
| 4 | **HIGH** | `model.py:16` | `whisper-tiny.en` hardcoded — English-only, poor alignment |
| 5 | **HIGH** | `trainer.py:283-284` | Gradient checkpointing set as bare attribute (inactive) |
| 6 | **HIGH** | `trainer.py:316-318` | StopIteration silently drops remaining training steps |
| 7 | **HIGH** | `trainer.py:344-348` | Public model push with no confirmation |
| 8 | **HIGH** | `utils.py:96-98` | Broken HF URL parsing (missing separator) |
| 9 | **MEDIUM** | `trainer.py:144,158` | `data_step` resume mismatch across config changes |
| 10 | **MEDIUM** | `preprocessing.py` | No validation of `.npy` output files |
| 11 | **MEDIUM** | `scraping.py:37,52-54` | Temp file race conditions |
| 12 | **LOW** | `test.py:87-99` | Beep injection should be configurable/removable |
| 13 | **LOW** | Entire codebase | Zero unit tests, zero integration tests |

---

## Recommendations for OmniVoxtral

1. **Fix loss_weights immediately** — This is the single highest-impact change. Wire `config.loss_weights` into `compute_loss()` with per-token-type masking.
2. **Replace whisper-tiny.en** — Use `openai/whisper-large-v3` for multilingual timestamps, or MMS for language detection + Whisper for alignment.
3. **Dual-stream architecture** — The single-stream assumption in `tokenizer/model.py` is fundamentally incompatible with duplex. Plan a clean rewrite.
4. **Add tests** — Zero coverage is unacceptable for a model that will be trained on expensive GPUs. Start with tokenizer roundtrip tests.
5. **Fix security issues** — `weights_only=True` for torch.load, private-by-default for HF push.
6. **Benchmark Mimi on Indic** — Before any architectural work, validate that Mimi can handle non-English speech at acceptable quality.
