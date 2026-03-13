# Literature Synthesis: Speech Language Models for OmniVoxtral

> Synthesis of key research with concrete positions for architectural decisions.
> Date: 2026-03-13 | Papers reviewed: 25+

---

## 1. Neural Audio Codecs

### Mimi (Kyutai, from Moshi paper arXiv:2410.00037)
- **Architecture:** SeaNet autoencoder + 8-layer Transformer bottleneck (8 heads, 250-frame context)
- **Specs:** 12.5Hz frame rate, Q=8 codebooks, 2048 codebook size, 1.1 kbps, 80ms latency
- **Innovation:** First RVQ level distilled from WavLM (semantic), remaining 7 levels are acoustic — "Split RVQ"
- **Quality:** PESQ ~2.45 on English benchmarks (below our 3.5 target)
- **Key finding:** Adversarial training improves subjective quality despite *degrading* PESQ/ViSQOL — PESQ alone is insufficient as a quality metric for neural codecs
- **Untested on Indic languages**

### EnCodec (Meta, arXiv:2210.13438)
- Multi-scale STFT discriminator, PESQ ~2.44 (comparable to Mimi)
- Not designed for streaming
- At 24kHz/6kbps: competitive quality but higher bitrate than Mimi

### DAC (Descript, arXiv:2306.06546)
- PESQ ~1.57 in some benchmarks (bitrate-dependent — higher bitrates much better)
- Higher absolute quality at 44.1kHz but not designed for speech LM integration
- No semantic distillation

### DM-Codec (arXiv:2409.17272)
- Reports ViSQOL 3.18 (above Mimi's 2.86)
- Uses disentangled semantic + acoustic codebooks with language model refinement
- Promising but newer, less validated

### SpeechTokenizer (arXiv:2308.16692)
- Semantic distillation from HuBERT into first codebook (like Mimi, independently discovered)
- 8 codebooks at 50Hz
- Validated on English TTS and voice conversion

**Position:** Mimi is our starting codec because:
1. Semantic distillation in first codebook enables Inner Monologue (critical for quality)
2. 80ms latency is duplex-compatible
3. Already integrated in Voxtral codebase
4. **But:** Must validate on Indic speech. PESQ <3.0 on English is concerning. If Indic quality is unacceptable, consider DM-Codec or fine-tuning Mimi on Indic data.

**Action:** Codec benchmark on 22 Indic languages before any model work.

---

## 2. Speech Language Models

### Moshi (Kyutai, arXiv:2410.00037) — THE reference architecture

**Architecture:**
- Temporal Transformer: 32 layers, 32 heads, dim 4096 (initialized from Helium 7B)
- Depth Transformer: 6 layers, 16 heads, dim 1024 (models inter-codebook dependencies per timestep)
- Dual streams: user audio + model audio processed simultaneously
- Inner Monologue: text tokens as prefix to audio — "one of the most critical impacts on quality"

**Training (5 phases, 1016 H100 GPUs):**
1. Text pre-training (Helium 7B, 2T tokens)
2. Audio pre-training (7M hours, unsupervised + Whisper-transcribed)
3. Diarized post-training (speaker separation)
4. Duplex fine-tuning (Fisher corpus, 2000 hours real conversations)
5. Instruction fine-tuning (20K hours synthetic instructional)

**Loss weighting:** `alpha=100` for semantic/text tokens, `alpha=1` for acoustic tokens. This is the loss_weights bug — Moshi found this weighting CRITICAL.

**Latency:** 160ms theoretical (80ms Mimi frame + 80ms acoustic delay τ=2), 200ms practical on L4 GPU.

**Benchmarks:**
- WebQ accuracy: 9% (no Inner Monologue) → 26.6% (with Inner Monologue) — 3× improvement
- Fisher turn-taking statistics match human ground truth
- PESQ of generated speech: competitive with ground truth after duplex fine-tuning

**Limitations:** English only. No multilingual. No language switching. 1016 H100s for training.

### SpiRit-LM (Meta, arXiv:2402.05755)
- Llama 2 7B extended with speech tokens (HuBERT-based)
- Word-level interleaving with `[SPEECH]`/`[TEXT]` markers
- Two versions: Base (phonetic only) + Expressive (pitch + style units)
- 21K A100-80GB GPU hours to train
- Can do few-shot ASR, TTS, Speech Classification across modalities
- **Key insight:** Interleaved speech+text on a text LLM backbone works well
- **Limitation:** NOT duplex, NOT streaming. Single-stream only.

### AudioLM (Google, arXiv:2209.03143)
- Established the semantic-then-acoustic hierarchy
- Two-stage: predict semantic tokens (w2v-BERT) → predict acoustic tokens (SoundStream)
- **Relevance:** Mimi's semantic distillation follows this principle. First codebook = semantic, rest = acoustic.

### VALL-E (Microsoft, arXiv:2301.02111)
- Zero-shot TTS via codec LM with acoustic prompt conditioning
- Predicts first codebook autoregressively, remaining codebooks in parallel (NAR)
- **Relevance:** Parallel vs sequential RVQ prediction. Moshi chose Depth Transformer (sequential across codebooks per timestep). VALL-E shows parallel works for TTS but not for real-time streaming.

### dGSLM (Meta/Facebook, arXiv:2203.16502)
- Dialogue Generative Spoken Language Model
- Dual-channel: two parallel HuBERT streams for two speakers
- Edge Transformer handles cross-speaker attention
- **Key insight:** Dual-stream is essential for modeling turn-taking and overlaps
- Trained on Fisher + Switchboard (5K hours)
- **Limitation:** HuBERT tokens only, no text, no inner monologue

**Position:** Moshi's architecture is the proven blueprint. Specifically:
1. **Temporal + Depth** transformer split (not monolithic)
2. **Dual-stream** (not single-stream like current Voxtral)
3. **Inner Monologue** (text tokens for semantic grounding)
4. **Loss weighting** (100:1 semantic:acoustic ratio)

---

## 3. Multilingual Speech Models

### Whisper (OpenAI, arXiv:2212.04356)
- 680K hours, 98 languages, encoder-decoder
- **Thesis:** "Enough supervised data at scale beats any cleverness"
- Large-v3: 5M training hours. Indic WER ranges: Hindi ~12%, Tamil ~18%, Kannada ~22%, Telugu ~20%
- Language token conditioning: `<|kn|>` prefix tells model which language to expect
- **Key lesson:** Scale the data, keep the architecture simple. Language tokens work.

### Meta MMS (arXiv:2305.13516)
- 500K hours self-supervised across 1,400+ languages
- Average 32 hours per language (New Testament readings)
- 1B param wav2vec 2.0 with language-specific adapters (LoRA-style)
- Halved Whisper's WER on 54 FLEURS languages
- **Key lesson:** Self-supervised pre-training + language adapters = massive multilingual coverage
- **Key lesson:** 32 hours/language is ENOUGH for usable quality with self-supervised pre-training
- **Key lesson for OmniVoxtral:** Don't need thousands of hours per language if we use SSL pre-training

### SeamlessM4T (Meta, arXiv:2308.11596)
- 100 languages, unified S2T/S2S/T2S/T2T model
- SeamlessStreaming variant achieves real-time with chunked attention
- SeamlessExpressive preserves prosody, speaking rate, pauses
- w2v-BERT 2.0 encoder (600M params) as speech encoder
- **Key lesson:** A single unified model CAN handle all speech tasks across 100+ languages
- **Key lesson:** Prosody preservation is achievable in translation — should be easier in same-language generation

### IndicWhisper (AI4Bharat, arXiv:2407.11564)
- Fine-tuned Whisper-large-v3 on IndicVoices data
- 40-60% WER reduction on 12 Indic languages vs vanilla Whisper
- Achieved by: (1) language-specific fine-tuning, (2) text normalization for Indic scripts, (3) punctuation post-processing
- **Key lesson:** Indic languages need dedicated fine-tuning even on top of Whisper. Vanilla multilingual models underperform.

### MMS-Zeroshot (Meta, arXiv:2305.13516)
- Extends to 1,100+ languages with zero-shot adapters
- Adapter approach: shared encoder + per-language linear adapter (tiny, ~100K params each)
- **Key lesson:** Language adapters are cheap and effective. We can add new languages post-training.

**Position:** Multilingual strategy for OmniVoxtral:
1. **Language tokens** (Whisper's approach) for conditioning during generation
2. **Language adapters** (MMS approach) for per-language quality tuning
3. **50 hours minimum** per language of supervised speech for usable quality
4. **SSL pre-training** on unlabeled Indic audio to bootstrap representations
5. **IndicWhisper** or MMS for text timestamping instead of whisper-tiny.en

---

## 4. Duplex & Turn-Taking

### Fisher Corpus (LDC)
- 2,000 hours of real English telephone conversations
- Natural overlaps, backchannels ("uh-huh"), interruptions, cross-talk
- Moshi trained on this for duplex behavior — the gold standard
- Speaker diarization available (who spoke when)
- **Critical:** No Indian equivalent exists. This is the biggest data gap.

### Turn-Taking Research (Psycholinguistics)
- Average human turn-switching gap: 200-250ms (Stivers et al., 2009)
- Backchannels occur every 5-10 seconds in natural conversation
- Overlap rate: 5-15% of conversation time
- **Implication:** Model must generate responses with <250ms latency to feel natural

### dGSLM Turn-Taking Findings
- Fisher corpus statistics match model-generated turn-taking distributions
- Cross-attention between speaker channels is essential for interruption modeling
- Without dual-stream, model cannot learn when to yield vs. hold the floor

**Position:** For duplex capability:
1. **Fisher corpus** (2K hours) is essential for English duplex training
2. **No Indic equivalent exists** — biggest gap. Options: (a) diarize IndicVoices conversational portions, (b) scrape Indic podcast interviews from YouTube, (c) create synthetic Indic conversations
3. **Dual-stream is non-negotiable** — single-stream cannot model interruptions
4. **200ms latency target** — matches human turn-switching expectations

---

## 5. Code-Switching

### Code-Switching in India
- 250M+ people code-switch daily (Hinglish, Kanglish, Tanglish, etc.)
- 30-50% WER increase on code-switched vs monolingual speech (Pratap et al., 2023)
- Intra-sentential switching (mid-sentence) is most common and hardest
- Matrix language (dominant) + embedded language pattern

### HiACC (Hindi-Accented Code-switched Corpus)
- Hinglish spontaneous speech (adults + children)
- Natural code-switching patterns
- Academic license

### Hindi-Marathi Code-Switching Dataset
- 450 hours of natural Hindi-Marathi conversational speech
- Published by IITB
- Includes diarization and language labels

### IndicLID (AI4Bharat)
- Language identification for 24 Indian languages
- Can detect language at utterance level
- Useful for automatic language labeling of scraped data

**Position:** Code-switching is a first-class requirement:
1. Must handle Hinglish, Kanglish, Tanglish natively
2. Language token approach may need augmentation — mid-utterance `<|lang_switch:en|>` tokens
3. Training data must include natural code-switched speech (not synthetic)
4. HiACC + Hindi-Marathi corpus + YouTube Hinglish scraping = minimum viable code-switching data

---

## 6. Key Research Questions (with positions)

### Q1: Which codec for Indic speech?
**Position:** Start with Mimi, validate on 22 languages. If PESQ < 3.0 on >50% of languages, fine-tune Mimi on IndicVoices audio or evaluate DM-Codec.
**Evidence:** Mimi PESQ 2.45 on English (Moshi paper), DM-Codec ViSQOL 3.18 (DM-Codec paper). No Indic benchmarks exist for any codec.

### Q2: 4 quantizers or 8?
**Position:** Benchmark both. Moshi uses 8 for quality. Voxtral uses 4 for speed. If 4q quality is within 0.5 PESQ of 8q, keep 4 (2× faster training).
**Evidence:** Mimi paper shows diminishing returns after 4-5 codebooks. But first codebook (semantic) is disproportionately important.

### Q3: How to handle timestamps for Indic languages?
**Position:** Replace whisper-tiny.en with Whisper-large-v3 (multilingual). For languages where Whisper alignment is poor, use Montreal Forced Aligner with language-specific acoustic models.
**Evidence:** IndicWhisper shows 40-60% WER reduction with Indic fine-tuning. MFA supports custom acoustic models.

### Q4: How much Indic data is enough?
**Position:** 50 hours supervised per language minimum (MMS showed 32h sufficient). 500+ hours unsupervised for SSL pre-training. Conversational data is 10× more valuable than read speech.
**Evidence:** MMS (32h/language with SSL), Whisper (scale thesis), Fisher corpus (2Kh for duplex).

### Q5: Single model or language-family models?
**Position:** Single model with language adapters. Train shared backbone, add per-language LoRA adapters (~100K params each).
**Evidence:** MMS (1,100+ languages, shared encoder + adapters), SeamlessM4T (100 languages, single model).

### Q6: How to achieve duplex without Fisher-equivalent Indic data?
**Position:** Three-pronged approach: (1) Train duplex on Fisher (English), (2) Transfer turn-taking behavior to Indic via multilingual backbone, (3) Fine-tune on diarized Indic conversational data (IndicVoices + YouTube).
**Evidence:** Moshi's duplex behavior comes from Fisher training. Cross-lingual transfer of prosodic patterns is established in SeamlessExpressive.

---

## 7. Concrete Numbers for Planning

| Metric | Value | Source |
|--------|-------|--------|
| Mimi latency | 80ms per frame | Moshi paper |
| Moshi latency | 200ms practical | Moshi paper |
| Human turn gap | 200-250ms | Stivers et al. 2009 |
| Moshi training | 1,016 H100s | Moshi paper |
| SpiRit-LM training | 21K A100 hours | SpiRit-LM paper |
| MMS languages | 1,400+ | MMS paper |
| MMS hours/language | 32 (average) | MMS paper |
| Mimi PESQ (English) | 2.45 | Moshi paper |
| EnCodec PESQ | 2.44 | EnCodec paper |
| Inner Monologue WebQ | 9% → 26.6% | Moshi paper |
| IndicVoices hours | 12,000+ | IndicVoices paper |
| Fisher hours | 2,000 | LDC |
| Loss weight ratio | 100:10:1 (text:semantic:acoustic) | Moshi paper |
| Code-switch WER penalty | +30-50% | Pratap et al. 2023 |
| Whisper Hindi WER | ~12% | Whisper paper |
| Whisper Tamil WER | ~18% | Whisper paper |
