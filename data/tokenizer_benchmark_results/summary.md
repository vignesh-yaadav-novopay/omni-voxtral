# Mistral 7B Tokenizer Fertility — Indic Languages

Tokenizer: mistralai/Mistral-7B-v0.3
Vocab size: 32768

## Per-Language Results

| Language | Script | Samples | Fertility (mean) | Fertility (median) | Chars/Token | UNK Rate | Target | Status |
|----------|--------|---------|-----------------|-------------------|-------------|----------|--------|--------|
| Malayalam (mal) | Dravidian | 100 | 21.84 | 21.46 | 0.46 | 0.0000 | ≤3.0 | **FAIL** |
| Odia (ori) | Odia | 100 | 18.49 | 18.28 | 0.37 | 0.0000 | ≤3.5 | **FAIL** |
| Telugu (tel) | Dravidian | 100 | 12.70 | 12.52 | 0.61 | 0.0000 | ≤3.0 | **FAIL** |
| Punjabi (pan) | Gurmukhi | 100 | 12.38 | 12.20 | 0.42 | 0.0000 | ≤3.5 | **FAIL** |
| Gujarati (guj) | Gujarati | 100 | 12.28 | 12.39 | 0.48 | 0.0000 | ≤3.5 | **FAIL** |
| Kannada (kan) | Dravidian | 100 | 11.80 | 11.72 | 0.73 | 0.0000 | ≤3.0 | **FAIL** |
| Tamil (tam) | Dravidian | 100 | 10.40 | 10.26 | 0.88 | 0.0000 | ≤3.0 | **FAIL** |
| Assamese (asm) | Bengali | 100 | 8.75 | 8.75 | 0.75 | 0.0000 | ≤3.5 | **FAIL** |
| Bengali (ben) | Bengali | 100 | 7.51 | 7.45 | 0.89 | 0.0000 | ≤3.5 | **FAIL** |
| Marathi (mar) | Devanagari | 100 | 7.24 | 7.15 | 0.95 | 0.0000 | ≤2.5 | **FAIL** |
| Nepali (nep) | Devanagari | 100 | 7.17 | 7.07 | 0.95 | 0.0000 | ≤2.5 | **FAIL** |
| Hindi (hin) | Devanagari | 100 | 5.30 | 5.28 | 0.96 | 0.0000 | ≤2.5 | **FAIL** |
| Urdu (urd) | Perso-Arabic | 100 | 4.59 | 4.53 | 1.01 | 0.0000 | ≤3.5 | **FAIL** |

## Summary Statistics

- **Overall mean fertility:** 10.80 tokens/word
- **Overall median fertility:** 10.08 tokens/word
- **Mean UNK rate:** 0.0000
- **Mean chars/token:** 0.73

## Per-Script-Family

| Script Family | Languages | Mean Fertility | Mean Chars/Token |
|--------------|-----------|---------------|-----------------|
| Devanagari | hin, mar, nep, san, kok, doi, mai, brx | 6.57 | 0.95 |
| Bengali | ben, asm | 8.13 | 0.82 |
| Dravidian | tam, tel, kan, mal | 14.19 | 0.67 |
| Gurmukhi | pan | 12.38 | 0.42 |
| Gujarati | guj | 12.28 | 0.48 |
| Odia | ori | 18.49 | 0.37 |
| Perso-Arabic | urd, kas, snd | 4.59 | 1.01 |

## Recommendation

Languages with fertility > 3.5: **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Tamil, Telugu, Urdu**
These languages need dedicated tokenizer support (SentencePiece Unigram training).

Overall mean fertility (10.80) is high. **A multilingual tokenizer is strongly recommended.**