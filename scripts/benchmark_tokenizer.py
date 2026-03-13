"""Benchmark tokenizer fertility on 22 Indic languages.

Tests Mistral's 32K BPE tokenizer on Indic text from FLEURS transcripts.
Measures tokens/word (fertility) and unknown token rate.

Usage: uv run scripts/benchmark_tokenizer.py
"""

import os

import pandas as pd
import transformers as tr
from datasets import load_dataset
from pydantic_settings import BaseSettings
from tqdm import tqdm

# FLEURS codes for 13 available Indic languages
FLEURS_LANG_MAP = {
    "asm": "as_in", "ben": "bn_in", "guj": "gu_in", "hin": "hi_in",
    "kan": "kn_in", "mal": "ml_in", "mar": "mr_in", "nep": "ne_np",
    "ori": "or_in", "pan": "pa_in", "tam": "ta_in", "tel": "te_in",
    "urd": "ur_pk",
}

ALL_LANGUAGES = {
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mni": "Manipuri",
    "mar": "Marathi", "nep": "Nepali", "ori": "Odia", "pan": "Punjabi",
    "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}

# Script families for analysis
SCRIPT_FAMILIES = {
    "Devanagari": ["hin", "mar", "nep", "san", "kok", "doi", "mai", "brx"],
    "Bengali": ["ben", "asm"],
    "Dravidian": ["tam", "tel", "kan", "mal"],
    "Gurmukhi": ["pan"],
    "Gujarati": ["guj"],
    "Odia": ["ori"],
    "Perso-Arabic": ["urd", "kas", "snd"],
    "Other": ["mni", "sat"],
}


class TokenizerBenchmarkConfig(BaseSettings):
    output_path: str = "./data/tokenizer_benchmark_results"
    mistral_model: str = "mistralai/Mistral-7B-v0.3"
    max_samples: int = 100  # per language
    target_fertility: dict = {
        "Latin": 1.5,
        "Devanagari": 2.5,
        "Dravidian": 3.0,
        "Other": 3.5,
    }


def compute_fertility(tokenizer: tr.PreTrainedTokenizer, text: str) -> dict:
    """Compute tokens/word fertility for a text string."""
    words = text.split()
    if len(words) == 0:
        return {"tokens": 0, "words": 0, "fertility": 0.0, "chars_per_token": 0.0}

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(token_ids)
    num_words = len(words)
    num_chars = len(text)

    # Check for unknown tokens
    unk_id = tokenizer.unk_token_id
    num_unk = sum(1 for t in token_ids if t == unk_id) if unk_id is not None else 0

    return {
        "tokens": num_tokens,
        "words": num_words,
        "chars": num_chars,
        "fertility": num_tokens / num_words if num_words > 0 else 0.0,
        "chars_per_token": num_chars / num_tokens if num_tokens > 0 else 0.0,
        "unk_count": num_unk,
        "unk_rate": num_unk / num_tokens if num_tokens > 0 else 0.0,
    }


def get_fleurs_transcripts(lang_code: str, fleurs_code: str, max_samples: int) -> list[str]:
    """Get transcripts from FLEURS dataset (streaming)."""
    try:
        ds = load_dataset(
            "google/fleurs", fleurs_code, split="test",
            streaming=True, trust_remote_code=True,
        )
        transcripts = []
        for sample in ds:
            if len(transcripts) >= max_samples:
                break
            text = sample.get("transcription", "") or sample.get("raw_transcription", "")
            if text and len(text.strip()) > 10:
                transcripts.append(text.strip())
        return transcripts
    except Exception as e:
        print(f"  Failed to load FLEURS for {lang_code}: {e}")
        return []


def benchmark_tokenizer(config: TokenizerBenchmarkConfig) -> None:
    """Benchmark Mistral tokenizer on Indic languages."""
    os.makedirs(config.output_path, exist_ok=True)

    print(f"Loading tokenizer: {config.mistral_model}")
    tokenizer = tr.AutoTokenizer.from_pretrained(config.mistral_model)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"UNK token: {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

    all_results = []

    for lang_code, lang_name in tqdm(ALL_LANGUAGES.items(), desc="Languages"):
        if lang_code not in FLEURS_LANG_MAP:
            print(f"  {lang_name}: No FLEURS data, skipping")
            continue

        print(f"\n  {lang_name} ({lang_code})...")
        transcripts = get_fleurs_transcripts(
            lang_code, FLEURS_LANG_MAP[lang_code], config.max_samples
        )

        if not transcripts:
            print(f"  {lang_name}: No transcripts found")
            continue

        for text in transcripts:
            metrics = compute_fertility(tokenizer, text)
            metrics["language"] = lang_code
            metrics["language_name"] = lang_name
            metrics["text_preview"] = text[:80]
            all_results.append(metrics)

    if not all_results:
        print("No results collected!")
        return

    df = pd.DataFrame(all_results)

    # Save raw results
    csv_path = os.path.join(config.output_path, "raw_fertility.csv")
    df.to_csv(csv_path, index=False)

    # Per-language aggregation
    lang_stats = df.groupby(["language", "language_name"]).agg({
        "fertility": ["mean", "median", "std", "min", "max"],
        "chars_per_token": ["mean"],
        "unk_rate": ["mean"],
        "tokens": ["sum"],
        "words": ["sum"],
        "text_preview": ["count"],
    })
    lang_stats.columns = [
        "fertility_mean", "fertility_median", "fertility_std",
        "fertility_min", "fertility_max",
        "chars_per_token", "unk_rate",
        "total_tokens", "total_words", "n_samples",
    ]
    lang_stats = lang_stats.reset_index()

    # Determine script family for each language
    lang_to_family = {}
    for family, langs in SCRIPT_FAMILIES.items():
        for lang in langs:
            lang_to_family[lang] = family

    # Generate summary
    lines = ["# Mistral 7B Tokenizer Fertility — Indic Languages\n"]
    lines.append(f"Tokenizer: {config.mistral_model}")
    lines.append(f"Vocab size: {tokenizer.vocab_size}\n")

    lines.append("## Per-Language Results\n")
    lines.append("| Language | Script | Samples | Fertility (mean) | Fertility (median) | Chars/Token | UNK Rate | Target | Status |")
    lines.append("|----------|--------|---------|-----------------|-------------------|-------------|----------|--------|--------|")

    for _, row in lang_stats.sort_values("fertility_mean", ascending=False).iterrows():
        lang = row["language"]
        family = lang_to_family.get(lang, "Other")

        # Determine target
        if family in ("Dravidian",):
            target = 3.0
        elif family in ("Devanagari",):
            target = 2.5
        else:
            target = 3.5

        status = "PASS" if row["fertility_mean"] <= target else "FAIL"

        lines.append(
            f"| {row['language_name']} ({lang}) | {family} | {int(row['n_samples'])} | "
            f"{row['fertility_mean']:.2f} | {row['fertility_median']:.2f} | "
            f"{row['chars_per_token']:.2f} | {row['unk_rate']:.4f} | "
            f"≤{target:.1f} | **{status}** |"
        )

    # Overall stats
    lines.append(f"\n## Summary Statistics\n")
    lines.append(f"- **Overall mean fertility:** {df['fertility'].mean():.2f} tokens/word")
    lines.append(f"- **Overall median fertility:** {df['fertility'].median():.2f} tokens/word")
    lines.append(f"- **Mean UNK rate:** {df['unk_rate'].mean():.4f}")
    lines.append(f"- **Mean chars/token:** {df['chars_per_token'].mean():.2f}")

    # Per-script-family stats
    lines.append(f"\n## Per-Script-Family\n")
    lines.append("| Script Family | Languages | Mean Fertility | Mean Chars/Token |")
    lines.append("|--------------|-----------|---------------|-----------------|")

    for family, langs in SCRIPT_FAMILIES.items():
        family_data = df[df["language"].isin(langs)]
        if len(family_data) > 0:
            lines.append(
                f"| {family} | {', '.join(langs)} | "
                f"{family_data['fertility'].mean():.2f} | "
                f"{family_data['chars_per_token'].mean():.2f} |"
            )

    # Recommendation
    lines.append("\n## Recommendation\n")
    high_fertility = lang_stats[lang_stats["fertility_mean"] > 3.5]
    if len(high_fertility) > 0:
        bad_langs = ", ".join(high_fertility["language_name"].tolist())
        lines.append(f"Languages with fertility > 3.5: **{bad_langs}**")
        lines.append("These languages need dedicated tokenizer support (SentencePiece Unigram training).\n")
    else:
        lines.append("All languages within acceptable fertility range.\n")

    mean_fertility = df["fertility"].mean()
    if mean_fertility > 3.0:
        lines.append(f"Overall mean fertility ({mean_fertility:.2f}) is high. **A multilingual tokenizer is strongly recommended.**")
    elif mean_fertility > 2.0:
        lines.append(f"Overall mean fertility ({mean_fertility:.2f}) is moderate. A multilingual tokenizer would improve efficiency.")
    else:
        lines.append(f"Overall mean fertility ({mean_fertility:.2f}) is acceptable. Mistral's tokenizer may be sufficient.")

    summary_text = "\n".join(lines)
    summary_path = os.path.join(config.output_path, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"\nResults saved to {config.output_path}/")


if __name__ == "__main__":
    config = TokenizerBenchmarkConfig()
    benchmark_tokenizer(config)
