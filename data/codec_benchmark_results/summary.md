# Mimi Codec Benchmark Results — Indic Languages

Date: 2026-03-13 05:52


## 4 Quantizers

| Language | Samples | PESQ | SI-SNR (dB) | F0 Corr | Energy Corr | Spk Rate | Enc (ms) | Dec (ms) | Verdict |
|----------|---------|------|-------------|---------|-------------|----------|----------|----------|---------|
| Assamese (asm) | 10 | 1.52 | -4.2 | 0.903 | 0.975 | 1.018 | 326 | 26 | **FAIL** |
| Bengali (ben) | 10 | 1.61 | -4.9 | 0.938 | 0.976 | 1.053 | 31 | 24 | **FAIL** |
| Bodo (brx) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 18 | 17 | **FAIL** |
| Dogri (doi) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Gujarati (guj) | 10 | 1.58 | -6.1 | 0.971 | 0.967 | 0.969 | 26 | 21 | **FAIL** |
| Hindi (hin) | 10 | 1.50 | -2.7 | 0.959 | 0.977 | 1.015 | 28 | 23 | **FAIL** |
| Kannada (kan) | 10 | 1.45 | -5.2 | 0.932 | 0.965 | 0.772 | 24 | 22 | **FAIL** |
| Kashmiri (kas) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Konkani (kok) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Maithili (mai) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Malayalam (mal) | 10 | 1.28 | -7.8 | 0.935 | 0.947 | 1.009 | 27 | 23 | **FAIL** |
| Marathi (mar) | 10 | 1.46 | -7.6 | 0.902 | 0.971 | 1.050 | 23 | 19 | **FAIL** |
| Manipuri (mni) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Nepali (nep) | 10 | 1.45 | -11.3 | 0.699 | 0.976 | 2.000 | 21 | 20 | **FAIL** |
| Odia (ori) | 10 | 1.18 | -10.4 | 0.874 | 0.929 | 0.868 | 20 | 19 | **FAIL** |
| Punjabi (pan) | 10 | 1.39 | -7.2 | 0.800 | 0.960 | 1.079 | 21 | 18 | **FAIL** |
| Sanskrit (san) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Santali (sat) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Sindhi (snd) | 3 | 1.57 | 14.0 | 0.000 | 0.632 | 1.000 | 12 | 13 | **FAIL** |
| Tamil (tam) | 10 | 1.56 | -2.0 | 0.870 | 0.972 | 1.039 | 24 | 22 | **FAIL** |
| Telugu (tel) | 10 | 1.46 | -4.8 | 0.886 | 0.967 | 0.981 | 22 | 21 | **FAIL** |
| Urdu (urd) | 10 | 1.27 | -7.4 | 0.933 | 0.937 | 1.029 | 19 | 19 | **FAIL** |

**Summary: 0/22 languages PASS (0%)**

> **RECOMMENDATION:** Mimi may not be suitable. Evaluate DM-Codec or train new codec.


## 8 Quantizers

| Language | Samples | PESQ | SI-SNR (dB) | F0 Corr | Energy Corr | Spk Rate | Enc (ms) | Dec (ms) | Verdict |
|----------|---------|------|-------------|---------|-------------|----------|----------|----------|---------|
| Assamese (asm) | 10 | 1.92 | 2.9 | 0.949 | 0.989 | 1.051 | 14 | 17 | **FAIL** |
| Bengali (ben) | 10 | 2.01 | 2.4 | 0.881 | 0.988 | 1.074 | 15 | 18 | **FAIL** |
| Bodo (brx) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Dogri (doi) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Gujarati (guj) | 10 | 2.16 | 2.1 | 0.987 | 0.985 | 1.038 | 14 | 17 | **FAIL** |
| Hindi (hin) | 10 | 1.98 | 3.8 | 0.973 | 0.990 | 1.062 | 15 | 18 | **FAIL** |
| Kannada (kan) | 10 | 1.84 | 2.1 | 0.942 | 0.986 | 0.990 | 14 | 17 | **FAIL** |
| Kashmiri (kas) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Konkani (kok) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Maithili (mai) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Malayalam (mal) | 10 | 1.60 | -0.7 | 0.978 | 0.976 | 1.038 | 15 | 18 | **FAIL** |
| Marathi (mar) | 10 | 1.93 | 0.5 | 0.980 | 0.987 | 1.085 | 14 | 17 | **FAIL** |
| Manipuri (mni) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Nepali (nep) | 10 | 1.90 | -2.0 | 0.831 | 0.986 | 1.618 | 14 | 17 | **FAIL** |
| Odia (ori) | 10 | 1.41 | -2.6 | 0.918 | 0.961 | 1.058 | 14 | 16 | **FAIL** |
| Punjabi (pan) | 10 | 1.73 | 0.1 | 0.972 | 0.983 | 1.546 | 14 | 17 | **FAIL** |
| Sanskrit (san) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Santali (sat) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 13 | 13 | **MARGINAL** |
| Sindhi (snd) | 3 | 2.69 | 16.5 | 0.000 | 0.756 | 1.000 | 14 | 13 | **MARGINAL** |
| Tamil (tam) | 10 | 2.10 | 5.0 | 0.928 | 0.989 | 1.046 | 15 | 17 | **FAIL** |
| Telugu (tel) | 10 | 1.93 | 2.4 | 0.954 | 0.986 | 0.993 | 14 | 16 | **FAIL** |
| Urdu (urd) | 10 | 1.60 | 0.5 | 0.968 | 0.970 | 1.041 | 14 | 17 | **FAIL** |

**Summary: 0/22 languages PASS (0%)**

> **RECOMMENDATION:** Mimi may not be suitable. Evaluate DM-Codec or train new codec.


## 4q vs 8q Comparison

- **Assamese**: 4q=1.52, 8q=1.92, delta=+0.40
- **Bengali**: 4q=1.61, 8q=2.01, delta=+0.40
- **Bodo**: 4q=1.57, 8q=2.69, delta=+1.11
- **Dogri**: 4q=1.57, 8q=2.69, delta=+1.11
- **Gujarati**: 4q=1.58, 8q=2.16, delta=+0.58
- **Hindi**: 4q=1.50, 8q=1.98, delta=+0.48
- **Kannada**: 4q=1.45, 8q=1.84, delta=+0.39
- **Kashmiri**: 4q=1.57, 8q=2.69, delta=+1.11
- **Konkani**: 4q=1.57, 8q=2.69, delta=+1.11
- **Maithili**: 4q=1.57, 8q=2.69, delta=+1.11
- **Malayalam**: 4q=1.28, 8q=1.60, delta=+0.31
- **Marathi**: 4q=1.46, 8q=1.93, delta=+0.46
- **Manipuri**: 4q=1.57, 8q=2.69, delta=+1.11
- **Nepali**: 4q=1.45, 8q=1.90, delta=+0.44
- **Odia**: 4q=1.18, 8q=1.41, delta=+0.23
- **Punjabi**: 4q=1.39, 8q=1.73, delta=+0.34
- **Sanskrit**: 4q=1.57, 8q=2.69, delta=+1.11
- **Santali**: 4q=1.57, 8q=2.69, delta=+1.11
- **Sindhi**: 4q=1.57, 8q=2.69, delta=+1.11
- **Tamil**: 4q=1.56, 8q=2.10, delta=+0.54
- **Telugu**: 4q=1.46, 8q=1.93, delta=+0.47
- **Urdu**: 4q=1.27, 8q=1.60, delta=+0.32