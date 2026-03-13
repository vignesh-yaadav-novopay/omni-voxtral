# Dataset Inventory for OmniVoxtral

> Comprehensive catalog of available speech datasets for training OmniVoxtral.
> Date: 2026-03-13 | Focus: 22 scheduled Indic languages + English duplex

---

## Target Languages (22 Constitutional Scheduled Languages)

| # | Language | Script | ISO 639-3 | Family | Speakers (M) |
|---|----------|--------|-----------|--------|---------------|
| 1 | Assamese | Bengali | asm | Indo-Aryan | 15 |
| 2 | Bengali | Bengali | ben | Indo-Aryan | 230 |
| 3 | Bodo | Devanagari | brx | Sino-Tibetan | 1.5 |
| 4 | Dogri | Devanagari | doi | Indo-Aryan | 3 |
| 5 | Gujarati | Gujarati | guj | Indo-Aryan | 55 |
| 6 | Hindi | Devanagari | hin | Indo-Aryan | 600 |
| 7 | Kannada | Kannada | kan | Dravidian | 45 |
| 8 | Kashmiri | Perso-Arabic | kas | Indo-Aryan | 7 |
| 9 | Konkani | Devanagari | kok | Indo-Aryan | 2.5 |
| 10 | Maithili | Devanagari | mai | Indo-Aryan | 34 |
| 11 | Malayalam | Malayalam | mal | Dravidian | 38 |
| 12 | Manipuri (Meitei) | Meitei | mni | Sino-Tibetan | 1.8 |
| 13 | Marathi | Devanagari | mar | Indo-Aryan | 83 |
| 14 | Nepali | Devanagari | nep | Indo-Aryan | 16 |
| 15 | Odia | Odia | ori | Indo-Aryan | 38 |
| 16 | Punjabi | Gurmukhi | pan | Indo-Aryan | 113 |
| 17 | Sanskrit | Devanagari | san | Indo-Aryan | <0.1 |
| 18 | Santali | Ol Chiki | sat | Austroasiatic | 7.5 |
| 19 | Sindhi | Perso-Arabic | snd | Indo-Aryan | 32 |
| 20 | Tamil | Tamil | tam | Dravidian | 75 |
| 21 | Telugu | Telugu | tel | Dravidian | 82 |
| 22 | Urdu | Perso-Arabic | urd | Indo-Aryan | 230 |

---

## Primary Datasets

### 1. IndicVoices (AI4Bharat)
- **Languages:** All 22 scheduled + some non-scheduled
- **Hours:** 12,000+ total (~7,000 transcribed, ~5,000 untranscribed)
- **Type:** Read speech + Extempore + Conversational
- **Speakers:** 16,000+ diverse speakers
- **License:** CC-BY-4.0
- **Source:** `ai4bharat/IndicVoices` on HuggingFace
- **Quality:** High (studio/controlled recording environments)
- **Per-language:** Median ~122 hours (ranges: 30h for Bodo → 800h+ for Hindi)
- **Key strength:** Only dataset covering ALL 22 languages with conversational portions
- **Key weakness:** Most content is read speech, conversational portions are limited

### 2. IndicVoices-R (AI4Bharat, NeurIPS 2024)
- **Languages:** 22 Indic
- **Hours:** 1,704
- **Type:** Curated high-quality subset
- **Speakers:** 10,496
- **License:** CC-BY-4.0
- **Source:** HuggingFace
- **Quality:** Very High (manually verified, cleaned)
- **Use case:** Validation and fine-tuning benchmark

### 3. BhasaAnuvaad (AI4Bharat)
- **Languages:** 13 Indic (Hindi, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Urdu, Assamese, Nepali)
- **Hours:** 44,000+ total
- **Type:** Parallel speech-text, translation-aligned
- **License:** CC-BY-4.0
- **Source:** HuggingFace
- **Quality:** High
- **Key strength:** Largest single Indic speech corpus. Translation-aligned enables cross-lingual training.
- **Key weakness:** May be heavily read/studio speech. Need to verify conversational content.

### 4. Vaani (IISc Bangalore/Google)
- **Languages:** 13 Indian states (multilingual, not all 22)
- **Hours:** 31,000+
- **Type:** Conversational (telephone-style)
- **License:** Research use (verify before commercial)
- **Source:** IISc API / Google partnership
- **Quality:** High (conversational, natural)
- **Key strength:** CONVERSATIONAL data — rare and valuable for duplex
- **Key weakness:** May not cover all 22 languages. Access may be restricted.
- **Action needed:** Verify access method and language coverage

### 5. Mozilla Common Voice (v16.1)
- **Languages:** ~15 Indic (Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, Assamese, Urdu, Nepali, Santali, others)
- **Hours:** 1,000+ total across Indic languages
- **Type:** Read speech (sentence-level)
- **License:** CC0 (public domain)
- **Source:** `mozilla-foundation/common_voice_16_1` on HuggingFace
- **Quality:** Medium (crowd-sourced, variable recording quality)
- **Per-language estimate:** Hindi ~50h, Tamil ~30h, Bengali ~20h, others 1-15h

### 6. Fisher Corpus (LDC)
- **Languages:** English
- **Hours:** 2,000
- **Type:** Duplex telephone conversations
- **License:** LDC License (academic/research, paid)
- **Source:** Linguistic Data Consortium (LDC2004T19, LDC2005T19)
- **Quality:** Gold standard for duplex
- **Key strength:** ONLY large-scale duplex conversational dataset. Moshi trained on this.
- **Key weakness:** English only. Requires LDC membership (~$600/year academic).
- **ESSENTIAL for duplex training**

### 7. FLEURS (Google)
- **Languages:** 102 languages including ~15 Indic
- **Hours:** ~200 total (~12h per language)
- **Type:** Read speech (Wikipedia sentences)
- **License:** CC-BY-4.0
- **Source:** `google/fleurs` on HuggingFace
- **Quality:** High (professional recording)
- **Use case:** Evaluation benchmark, not training (too small per language)

---

## Secondary Datasets

### 8. OpenSLR Datasets (Various)
| OpenSLR ID | Language | Hours | Type | License |
|------------|----------|-------|------|---------|
| SLR63 | Hindi (Large) | 1,100+ | Read | CC-BY-4.0 |
| SLR65 | Tamil | 40 | Read | CC-BY-4.0 |
| SLR66 | Telugu | 40 | Read | CC-BY-4.0 |
| SLR64 | Malayalam | 30 | Read | CC-BY-4.0 |
| SLR79 | Gujarati | 40 | Read | CC-BY-4.0 |
| SLR80 | Marathi | 40 | Read | CC-BY-4.0 |
| SLR118 | Bengali | 24 | Read | CC-BY-4.0 |

### 9. YouTube (Existing Pipeline)
- **Languages:** English (current searches.txt)
- **Hours:** 8,000+ (estimated from existing pipeline output)
- **Type:** Podcast conversational
- **License:** Fair use (research)
- **Quality:** Variable (includes background music, ads, multi-speaker)
- **Source:** Existing Voxtral pipeline (`data/indexing.py` → `data/scraping.py`)

### 10. YouTube (Indic Extension)
- **Languages:** All 22 (via Indic search terms)
- **Hours:** Estimated 5,000-10,000+ (dependent on search term quality)
- **Type:** Conversational (podcasts, interviews, panel discussions)
- **License:** Fair use (research)
- **Source:** Extended pipeline with Indic native-script search terms
- **Action needed:** Create Indic search terms file for all 22 languages

---

## Code-Switching Datasets

### 11. HiACC (Hindi-Accented Code-switched Corpus)
- **Languages:** Hindi-English (Hinglish)
- **Hours:** ~100 (estimated)
- **Type:** Spontaneous code-switched speech
- **License:** Academic
- **Key value:** Natural Hinglish patterns

### 12. Hindi-Marathi Code-Switching (IITB)
- **Languages:** Hindi-Marathi
- **Hours:** 450
- **Type:** Conversational code-switching
- **License:** Academic
- **Key value:** Rare non-English code-switching data

### 13. MUCS 2021 Shared Task Data
- **Languages:** Hindi-English, Bengali-English, Telugu-English
- **Hours:** ~50 per pair
- **Type:** Read + spontaneous code-switched
- **License:** Research

---

## Self-Supervised / Unlabeled Audio

### 14. VoxPopuli (Meta)
- **Languages:** 23 European + transcribed portions
- **Hours:** 400K+ (untranscribed)
- **Relevance:** Methodology reference for SSL on large unlabeled corpora

### 15. Libri-Light (Meta)
- **Languages:** English
- **Hours:** 60K (unlabeled audio books)
- **Relevance:** English SSL pre-training data

### 16. All India Radio Archives
- **Languages:** All 22+ in broadcast format
- **Hours:** Unknown (potentially massive)
- **Status:** Access uncertain, may require RTI request or partnership
- **Quality:** Professional broadcast quality

---

## Hours Summary by Language

| Language | IndicVoices | BhasaAnuvaad | CommonVoice | OpenSLR | YouTube (est.) | Total Est. |
|----------|-------------|--------------|-------------|---------|---------------|------------|
| Hindi | 800+ | 15,000+ | 50+ | 1,100+ | 2,000+ | **19,000+** |
| Bengali | 500+ | 5,000+ | 20+ | 24 | 500+ | **6,000+** |
| Tamil | 400+ | 4,000+ | 30+ | 40 | 500+ | **5,000+** |
| Telugu | 400+ | 4,000+ | 20+ | 40 | 500+ | **5,000+** |
| Marathi | 400+ | 3,000+ | 15+ | 40 | 300+ | **3,800+** |
| Gujarati | 300+ | 2,000+ | 10+ | 40 | 200+ | **2,600+** |
| Kannada | 300+ | 2,000+ | 15+ | — | 300+ | **2,600+** |
| Malayalam | 300+ | 2,000+ | 10+ | 30 | 200+ | **2,500+** |
| Punjabi | 200+ | 1,500+ | 10+ | — | 200+ | **1,900+** |
| Odia | 200+ | 1,000+ | 5+ | — | 100+ | **1,300+** |
| Urdu | 200+ | 1,500+ | 10+ | — | 300+ | **2,000+** |
| Assamese | 100+ | 500+ | 5+ | — | 50+ | **700+** |
| Nepali | 100+ | 500+ | 15+ | — | 100+ | **700+** |
| Manipuri | 50+ | — | 1+ | — | 20+ | **70+** |
| Konkani | 50+ | — | 1+ | — | 20+ | **70+** |
| Maithili | 50+ | — | — | — | 20+ | **70+** |
| Kashmiri | 50+ | — | — | — | 10+ | **60+** |
| Sindhi | 50+ | — | — | — | 20+ | **70+** |
| Dogri | 30+ | — | — | — | 10+ | **40+** |
| Bodo | 30+ | — | — | — | 5+ | **35+** |
| Sanskrit | 30+ | — | — | — | 10+ | **40+** |
| Santali | 30+ | — | 2+ | — | 5+ | **37+** |

**Estimated total: ~90,000-100,000+ hours across all sources**

---

## Critical Gaps

### Gap 1: Indic Conversational/Duplex Data
**Problem:** Almost all Indic datasets are read speech or semi-structured. No Fisher-equivalent exists for any Indic language.
**Impact:** Cannot train duplex turn-taking behavior directly on Indic data.
**Mitigation:**
1. Use Fisher (English) for duplex training, transfer to Indic via multilingual backbone
2. Diarize IndicVoices conversational portions for partial Indic duplex data
3. YouTube Indic scraping prioritizing podcast interviews (natural turn-taking)
4. Vaani (IISc) — verify if conversational portions are accessible

### Gap 2: Code-Switching Data
**Problem:** <600 hours total across all code-switching datasets. Hinglish has ~100h, other pairs have less.
**Impact:** Model may struggle with mid-sentence language switching (the most common pattern in Indian speech).
**Mitigation:**
1. YouTube scraping with code-switching-specific search terms ("hinglish podcast", "tamil english mix")
2. Data augmentation: splice monolingual segments with code-switching transition patterns
3. Prioritize HiACC + Hindi-Marathi corpus for initial training

### Gap 3: Low-Resource Languages
**Problem:** Bodo, Santali, Dogri, Maithili, Manipuri each have <100 hours.
**Impact:** Insufficient for standalone training. May produce unintelligible output.
**Mitigation:**
1. SSL pre-training on unlabeled audio (follow MMS: 32h supervised sufficient with SSL)
2. Language-family transfer learning (Bodo→Assamese family, Santali→Bengali proximity)
3. Aggressive YouTube scraping for these languages
4. Community partnerships for data collection

### Gap 4: Emotional/Expressive Speech
**Problem:** Most datasets are neutral/read speech. Natural conversation has emotion, emphasis, sarcasm.
**Impact:** Model may sound flat/robotic, defeating the purpose of end-to-end speech models.
**Mitigation:**
1. YouTube podcasts naturally contain emotional speech
2. Prioritize conversational/extempore portions of IndicVoices
3. Consider IEMOCAP or MSP-Podcast for emotional speech pre-training (English)

---

## Data Priority Matrix

| Priority | Dataset | Why |
|----------|---------|-----|
| **P0 (Must have)** | IndicVoices (full) | Only dataset covering all 22 languages |
| **P0** | Fisher Corpus | Only duplex conversational dataset (English) |
| **P0** | YouTube Indic (scrape) | Conversational Indic data at scale |
| **P1 (Should have)** | BhasaAnuvaad | Massive hours, cross-lingual alignment |
| **P1** | Vaani | Real conversational Indic data (if accessible) |
| **P1** | IndicVoices-R | High-quality validation benchmark |
| **P1** | Common Voice | CC0 license, reliable quality |
| **P2 (Nice to have)** | HiACC | Code-switching patterns |
| **P2** | Hindi-Marathi CS | Non-English code-switching |
| **P2** | OpenSLR datasets | Additional per-language hours |
| **P2** | FLEURS | Evaluation benchmark |
| **P3 (Future)** | VoxPopuli | SSL pre-training methodology |
| **P3** | All India Radio | Professional broadcast (if accessible) |

---

## Download & Processing Plan

### Phase 1 (Codec Benchmark — 10 samples/language)
- Source: IndicVoices + FLEURS (small, easy to download)
- Total: 220 samples × 10s = ~37 minutes
- No GPU needed for download

### Phase 2 (Initial Training — 50h/language target)
- Source: IndicVoices (full download), Common Voice, OpenSLR
- Total: ~5,000-10,000 hours
- Processing: GPU tokenization with extended pipeline

### Phase 3 (Scale Training — all available data)
- Source: All P0+P1 datasets + YouTube Indic scraping
- Total: ~50,000-100,000 hours
- Processing: Distributed GPU tokenization (Ray or multi-node)
