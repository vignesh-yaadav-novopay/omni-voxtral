# Interview Transcript

## Round 1 — Architectural decisions

### Q1: Option A vs B — what goes into the training loss?
**Selected:** A: Source-language IM, English as sidecar (Recommended)

Rationale: Text tokens = transcript in spoken language. Translation stored only as metadata. Preserves Inner Monologue causal alignment; matches Moshi/Voxtral. Indic-Indic duplex works.

### Q2: What to do with existing 57,668 corrupted YouTube .npy tokens?
**Selected:** Discard, re-tokenize from chunks (Recommended)

Cost: ~200 GPU-hr Whisper+Mimi + ~30 GPU-hr diarization. Cleanest training signal.

### Q3: Commercial / licensing posture?
**Selected:** Research / non-commercial only — MMS is fine

Can freely use Meta MMS (CC-BY-NC 4.0) for LID and ASR fallback. Covers all 22 Indic languages.

### Q4: Phase ordering — which ships first so training can resume quickly?
**Selected:** Phase 1: Tokenizer fixes only (Recommended)

Defects 2, 3, 5 — SentencePiece + task=translate + metadata sidecars. Re-tokenize existing 20s chunks WITHOUT re-chunking. Unblocks training in days. VAD/diarization come later.

---

## Round 2 — Implementation details

### Q5: Primary ASR choice?
**Selected:** faster-whisper-large-v3 (Recommended)

Battle-tested for Indic via Vistaar/IndicWhisper benchmarks. Built-in Silero VAD. ~3GB fp16. Use OpenAI Whisper for translation in a 2nd pass.

### Q6: Demucs source separation default?
**Selected:** Off by default, opt-in flag (Recommended)

Most YouTube podcast intro music is short. Demucs adds ~3-5 min/hr preprocessing cost and can over-strip. Enable per-source via config (e.g., music-heavy channels).

### Q7: v2 token storage layout?
**Selected:** `data/tokens_v2/{source}/{lang}/{shard}/file.npy + .meta.json` (Recommended)

Source-aware layout (yt/fleurs/iv). Per-language directory for sampling. JSON sidecar per .npy file. Old `data/tokens/` stays untouched until cutover.

### Q8: Test framework?
**Selected:** pytest + integration tests on every component (Recommended)

`tests/` directory, pytest as runner, fixtures for synthetic audio. Each new module ships with unit + 1 integration test. ~1.5x dev time, catches regressions.

---

## Implicit decisions (carried forward from prior conversation)

- **Goal model:** preserves prosody, learns when to speak vs listen (interruptions), produces intelligible Indic speech.
- **YouTube corpus is primary target** — 57,668 chunks at `data/chunks_indic_yt/`.
- **Hardware:** single A10G for preprocessing, 4×A10G DDP for training.
- **Whisper-diarization repo (Ashraf 2024) stage ordering** is the integration blueprint, with pyannote-3.1 substituted for NeMo TitaNet.
- **Mimi treated as mono** — stereo handled via per-channel separate Mimi-mono streams (Moshi pattern).
- **Silero VAD v6 per-channel** with `min_speech_duration_ms=1000, min_silence_duration_ms=500, speech_pad_ms=150, max_speech_duration_s=20`.
- **mms-lid-2048 as primary LID + Whisper-large-v3 LID as confidence tie-breaker**.
- **MMS-1B-all** for ASR on the 7 Whisper-unsupported Indic languages.
- **Existing SentencePiece model** (`data/tokenizer/omnivoxtral_sp.model`, 65K vocab, fertility 1.91) is the canonical text tokenizer — no retraining in this rebuild.
- **Preserve Moshi's per-frame inner-monologue prefix pattern** for language tags.
