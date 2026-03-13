"""Unified Indic dataset loader for OmniVoxtral training data.

Loads audio + transcripts from multiple sources into a common schema.
Supports: IndicVoices, FLEURS, Mozilla Common Voice (streaming mode).

Usage: uv run scripts/load_indic_datasets.py
"""

import os
from dataclasses import asdict, dataclass

import dotenv
import pandas as pd

dotenv.load_dotenv()
from datasets import load_dataset
from pydantic_settings import BaseSettings
from tqdm import tqdm


@dataclass
class AudioSample:
    """Unified schema for all audio sources."""

    source: str  # "indicvoices", "fleurs", "commonvoice", "youtube"
    language_code: str  # ISO 639-3: "hin", "tam", "kan", etc.
    language_name: str
    transcript: str | None
    duration_sec: float
    speaker_id: str | None
    split: str  # "train", "test", "validation"
    sample_id: str


# All 22 scheduled languages
ALL_LANGUAGES = {
    "asm": "Assamese",
    "ben": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "kas": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "mal": "Malayalam",
    "mni": "Manipuri",
    "mar": "Marathi",
    "nep": "Nepali",
    "ori": "Odia",
    "pan": "Punjabi",
    "san": "Sanskrit",
    "sat": "Santali",
    "snd": "Sindhi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}

# FLEURS language codes (13 available)
FLEURS_LANG_MAP = {
    "asm": "as_in",
    "ben": "bn_in",
    "guj": "gu_in",
    "hin": "hi_in",
    "kan": "kn_in",
    "mal": "ml_in",
    "mar": "mr_in",
    "nep": "ne_np",
    "ori": "or_in",
    "pan": "pa_in",
    "tam": "ta_in",
    "tel": "te_in",
    "urd": "ur_pk",
}

# Common Voice language codes (approximate mapping)
CV_LANG_MAP = {
    "asm": "as",
    "ben": "bn",
    "guj": "gu",
    "hin": "hi",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "nep": "ne-NP",
    "ori": "or",
    "pan": "pa-IN",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}

# IndicVoices language codes (lowercase config names per HF Hub)
IV_LANG_MAP = {
    "asm": "assamese",
    "ben": "bengali",
    "brx": "bodo",
    "doi": "dogri",
    "guj": "gujarati",
    "hin": "hindi",
    "kan": "kannada",
    "kas": "kashmiri",
    "kok": "konkani",
    "mai": "maithili",
    "mal": "malayalam",
    "mni": "manipuri",
    "mar": "marathi",
    "nep": "nepali",
    "ori": "odia",
    "pan": "punjabi",
    "san": "sanskrit",
    "sat": "santali",
    "snd": "sindhi",
    "tam": "tamil",
    "tel": "telugu",
    "urd": "urdu",
}


class DatasetLoaderConfig(BaseSettings):
    output_path: str = "./data/dataset_inventory"
    max_samples_per_lang: int = 500  # For inventory scan (not full download)
    sources: list[str] = ["fleurs", "commonvoice", "indicvoices"]
    splits: list[str] = ["train"]


def scan_fleurs(config: DatasetLoaderConfig) -> list[dict]:
    """Scan FLEURS for available Indic language data."""
    results = []

    for lang_code, fleurs_code in tqdm(FLEURS_LANG_MAP.items(), desc="FLEURS"):
        lang_name = ALL_LANGUAGES[lang_code]
        for split in config.splits:
            try:
                ds = load_dataset(
                    "google/fleurs",
                    fleurs_code,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )

                count = 0
                total_duration = 0.0
                for sample in ds:
                    if count >= config.max_samples_per_lang:
                        break
                    transcript = sample.get("transcription", "") or sample.get(
                        "raw_transcription", ""
                    )
                    # Estimate duration from audio array length
                    audio = sample.get("audio", {})
                    sr = audio.get("sampling_rate", 16000)
                    audio_array = audio.get("array")
                    duration = len(audio_array) / sr if audio_array is not None else 0

                    if transcript and len(transcript.strip()) > 5:
                        results.append(
                            asdict(
                                AudioSample(
                                    source="fleurs",
                                    language_code=lang_code,
                                    language_name=lang_name,
                                    transcript=transcript.strip(),
                                    duration_sec=round(duration, 2),
                                    speaker_id=str(sample.get("id", "")),
                                    split=split,
                                    sample_id=f"fleurs_{fleurs_code}_{count}",
                                )
                            )
                        )
                        total_duration += duration
                        count += 1

                print(
                    f"  FLEURS {lang_name} ({split}): {count} samples, "
                    f"{total_duration / 3600:.1f}h"
                )
            except Exception as e:
                print(f"  FLEURS {lang_name} ({split}): Failed - {e}")

    return results


def scan_commonvoice(config: DatasetLoaderConfig) -> list[dict]:
    """Scan Mozilla Common Voice for available Indic language data."""
    results = []

    for lang_code, cv_code in tqdm(CV_LANG_MAP.items(), desc="CommonVoice"):
        lang_name = ALL_LANGUAGES[lang_code]
        for split in config.splits:
            try:
                ds = load_dataset(
                    "mozilla-foundation/common_voice_16_1",
                    cv_code,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )

                count = 0
                total_duration = 0.0
                for sample in ds:
                    if count >= config.max_samples_per_lang:
                        break
                    transcript = sample.get("sentence", "")
                    audio = sample.get("audio", {})
                    sr = audio.get("sampling_rate", 48000)
                    audio_array = audio.get("array")
                    duration = len(audio_array) / sr if audio_array is not None else 0

                    if transcript and len(transcript.strip()) > 3:
                        results.append(
                            asdict(
                                AudioSample(
                                    source="commonvoice",
                                    language_code=lang_code,
                                    language_name=lang_name,
                                    transcript=transcript.strip(),
                                    duration_sec=round(duration, 2),
                                    speaker_id=sample.get("client_id", ""),
                                    split=split,
                                    sample_id=f"cv_{cv_code}_{count}",
                                )
                            )
                        )
                        total_duration += duration
                        count += 1

                print(
                    f"  CommonVoice {lang_name} ({split}): {count} samples, "
                    f"{total_duration / 3600:.1f}h"
                )
            except Exception as e:
                print(f"  CommonVoice {lang_name} ({split}): Failed - {e}")

    return results


def scan_indicvoices(config: DatasetLoaderConfig) -> list[dict]:
    """Scan IndicVoices for available data."""
    results = []

    for lang_code, lang_name in tqdm(IV_LANG_MAP.items(), desc="IndicVoices"):
        for split in config.splits:
            try:
                ds = load_dataset(
                    "ai4bharat/IndicVoices",
                    lang_name,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )

                count = 0
                total_duration = 0.0
                for sample in ds:
                    if count >= config.max_samples_per_lang:
                        break
                    # IndicVoices uses 'normalized' or 'verbatim' for transcript
                    transcript = (
                        sample.get("normalized", "")
                        or sample.get("verbatim", "")
                        or sample.get("transcript", "")
                        or sample.get("text", "")
                    )
                    # Duration may be a direct field
                    duration = sample.get("duration", 0.0)
                    if duration == 0:
                        audio = sample.get("audio", {})
                        sr = audio.get("sampling_rate", 16000)
                        audio_array = audio.get("array")
                        duration = (
                            len(audio_array) / sr if audio_array is not None else 0
                        )

                    results.append(
                        asdict(
                            AudioSample(
                                source="indicvoices",
                                language_code=lang_code,
                                language_name=ALL_LANGUAGES[lang_code],
                                transcript=transcript.strip() if transcript else None,
                                duration_sec=round(duration, 2),
                                speaker_id=sample.get("speaker_id", ""),
                                split=split,
                                sample_id=f"iv_{lang_code}_{count}",
                            )
                        )
                    )
                    total_duration += duration
                    count += 1

                print(
                    f"  IndicVoices {lang_name} ({split}): {count} samples, "
                    f"{total_duration / 3600:.1f}h"
                )
            except Exception as e:
                print(f"  IndicVoices {lang_name} ({split}): Failed - {e}")

    return results


def load_indic_datasets(config: DatasetLoaderConfig) -> None:
    """Scan all configured dataset sources and produce inventory."""
    os.makedirs(config.output_path, exist_ok=True)

    all_results = []

    if "fleurs" in config.sources:
        print("\n=== Scanning FLEURS ===")
        all_results.extend(scan_fleurs(config))

    if "commonvoice" in config.sources:
        print("\n=== Scanning Common Voice ===")
        all_results.extend(scan_commonvoice(config))

    if "indicvoices" in config.sources:
        print("\n=== Scanning IndicVoices ===")
        all_results.extend(scan_indicvoices(config))

    if not all_results:
        print("No results collected!")
        return

    df = pd.DataFrame(all_results)

    # Save raw inventory
    csv_path = os.path.join(config.output_path, "raw_inventory.csv")
    df.to_csv(csv_path, index=False)

    # Per-language, per-source summary
    summary = (
        df.groupby(["language_code", "language_name", "source"])
        .agg(
            samples=("sample_id", "count"),
            total_hours=("duration_sec", lambda x: x.sum() / 3600),
            has_transcript=("transcript", lambda x: x.notna().sum()),
        )
        .reset_index()
    )

    summary_path = os.path.join(config.output_path, "summary.csv")
    summary.to_csv(summary_path, index=False)

    # Print summary table
    print("\n" + "=" * 80)
    print("DATASET INVENTORY SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Language':<15} {'Source':<15} {'Samples':<10} {'Hours':<10} {'Transcribed':<12}"
    )
    print("-" * 62)

    for _, row in summary.sort_values(
        ["language_code", "source"]
    ).iterrows():
        print(
            f"{row['language_name']:<15} {row['source']:<15} "
            f"{row['samples']:<10} {row['total_hours']:.1f}h{'':<5} "
            f"{row['has_transcript']}"
        )

    # Overall stats
    total_samples = len(df)
    total_hours = df["duration_sec"].sum() / 3600
    total_languages = df["language_code"].nunique()
    total_sources = df["source"].nunique()

    print(f"\nTotal: {total_samples} samples, {total_hours:.1f} hours, "
          f"{total_languages} languages, {total_sources} sources")

    # Save markdown summary
    md_lines = ["# Indic Dataset Inventory\n"]
    md_lines.append(f"Date: 2026-03-13\n")
    md_lines.append(f"Total: {total_samples} samples, {total_hours:.1f} hours, "
                    f"{total_languages} languages\n")
    md_lines.append("| Language | Source | Samples | Hours | Transcribed |")
    md_lines.append("|----------|--------|---------|-------|-------------|")
    for _, row in summary.sort_values(["language_code", "source"]).iterrows():
        md_lines.append(
            f"| {row['language_name']} ({row['language_code']}) | "
            f"{row['source']} | {row['samples']} | "
            f"{row['total_hours']:.1f}h | {row['has_transcript']} |"
        )

    md_path = os.path.join(config.output_path, "inventory.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nInventory saved to {config.output_path}/")


if __name__ == "__main__":
    config = DatasetLoaderConfig()
    load_indic_datasets(config)
