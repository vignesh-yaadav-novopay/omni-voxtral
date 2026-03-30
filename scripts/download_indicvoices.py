"""Download IndicVoices audio for specified languages.

Streams from HuggingFace and saves .flac files for preprocessing.

Usage:
    LANG=kannada MAX_SAMPLES=2000 uv run python scripts/download_indicvoices.py
    LANG=malayalam,marathi,gujarati MAX_SAMPLES=2000 uv run python scripts/download_indicvoices.py
"""

import os
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from pydantic_settings import BaseSettings


class DownloadConfig(BaseSettings):
    lang: str = "kannada"  # comma-separated: "kannada,malayalam,marathi"
    max_samples: int = 2000
    output_path: str = "./data/chunks_indic"
    min_duration: float = 1.0  # skip very short utterances (<1s)
    max_duration: float = 30.0  # skip very long utterances (>30s)


def download_language(lang_name: str, config: DownloadConfig) -> int:
    """Download audio for a single language. Returns count of saved files."""
    # Map full name to 2-letter code
    name_to_code = {
        "assamese": "as", "bengali": "bn", "bodo": "brx", "dogri": "doi",
        "gujarati": "gu", "hindi": "hi", "kannada": "kn", "kashmiri": "ks",
        "konkani": "kok", "maithili": "mai", "malayalam": "ml", "manipuri": "mni",
        "marathi": "mr", "nepali": "ne", "odia": "or", "punjabi": "pa",
        "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
        "telugu": "te", "urdu": "ur",
    }
    code = name_to_code.get(lang_name, lang_name[:2])
    out_dir = os.path.join(config.output_path, code)
    os.makedirs(out_dir, exist_ok=True)

    # Check existing files
    existing = len([f for f in os.listdir(out_dir) if f.endswith(".flac")])
    if existing >= config.max_samples:
        print(f"  {lang_name}: already have {existing} files, skipping")
        return existing

    print(f"  Downloading {lang_name} (target: {config.max_samples}, existing: {existing})...")
    ds = load_dataset(
        "ai4bharat/IndicVoices", lang_name,
        split="train", streaming=True, trust_remote_code=True,
    )

    saved = existing
    skipped = 0
    for sample in tqdm(ds, desc=f"  {lang_name}", total=config.max_samples):
        if saved >= config.max_samples:
            break

        # Get audio from audio_filepath field
        audio_data = sample.get("audio_filepath", sample.get("audio", {}))
        if isinstance(audio_data, dict):
            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", 16000)
        else:
            skipped += 1
            continue

        if array is None:
            skipped += 1
            continue

        array = np.array(array)
        duration = len(array) / sr

        if duration < config.min_duration or duration > config.max_duration:
            skipped += 1
            continue

        # Save as flac
        filename = f"{code}_{saved:06d}.flac"
        filepath = os.path.join(out_dir, filename)
        sf.write(filepath, array, sr)
        saved += 1

    print(f"  {lang_name}: saved {saved} files, skipped {skipped}")
    return saved


def main():
    config = DownloadConfig()
    languages = [l.strip() for l in config.lang.split(",")]

    print(f"=== Downloading IndicVoices ===")
    print(f"  Languages: {languages}")
    print(f"  Max samples per lang: {config.max_samples}")
    print(f"  Output: {config.output_path}")
    print()

    total = 0
    for lang in languages:
        count = download_language(lang, config)
        total += count

    print(f"\n=== Done: {total} total files across {len(languages)} languages ===")


if __name__ == "__main__":
    main()
