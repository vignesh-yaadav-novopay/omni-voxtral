"""Download test audio samples for 22 Indic languages from FLEURS (streaming).

Saves 10 samples per language to data/indic_test_samples/{lang_code}/.
Uses streaming to avoid downloading the full dataset to disk.

Usage: uv run scripts/source_indic_audio.py
"""

import os

import torch
import torchaudio
from datasets import load_dataset
from pydantic_settings import BaseSettings
from tqdm import tqdm

# FLEURS language codes for our 22 Indic languages
FLEURS_LANG_MAP = {
    "asm": "as_in",  # Assamese
    "ben": "bn_in",  # Bengali
    "guj": "gu_in",  # Gujarati
    "hin": "hi_in",  # Hindi
    "kan": "kn_in",  # Kannada
    "mal": "ml_in",  # Malayalam
    "mar": "mr_in",  # Marathi
    "nep": "ne_np",  # Nepali
    "ori": "or_in",  # Odia
    "pan": "pa_in",  # Punjabi
    "tam": "ta_in",  # Tamil
    "tel": "te_in",  # Telugu
    "urd": "ur_pk",  # Urdu
}

# Languages not in FLEURS — create synthetic placeholders
NO_SOURCE_LANGS = ["brx", "doi", "kas", "kok", "mai", "mni", "san", "sat", "snd"]

# All 22 target languages
ALL_LANGUAGES = {
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mni": "Manipuri",
    "mar": "Marathi", "nep": "Nepali", "ori": "Odia", "pan": "Punjabi",
    "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}


class SourceConfig(BaseSettings):
    output_path: str = "./data/indic_test_samples"
    samples_per_language: int = 10
    min_duration_sec: float = 3.0
    max_duration_sec: float = 15.0
    target_sample_rate: int = 24000


def save_audio(waveform: torch.Tensor, sample_rate: int, path: str, target_sr: int = 24000) -> bool:
    """Save audio, resampling to target_sr if needed."""
    try:
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(path, waveform, sample_rate)
        return True
    except Exception as e:
        print(f"  Error saving {path}: {e}")
        return False


def source_from_fleurs(lang_code: str, fleurs_code: str, config: SourceConfig) -> int:
    """Download samples from FLEURS using streaming. Returns number saved."""
    output_dir = os.path.join(config.output_path, lang_code)
    os.makedirs(output_dir, exist_ok=True)

    try:
        ds = load_dataset(
            "google/fleurs",
            fleurs_code,
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed to load FLEURS for {lang_code} ({fleurs_code}): {e}")
        return 0

    saved = 0
    tried = 0
    for sample in ds:
        if saved >= config.samples_per_language:
            break
        tried += 1
        if tried > 200:  # Don't scan too many samples
            break

        audio = sample["audio"]
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sr = audio["sampling_rate"]
        duration = waveform.shape[-1] / sr

        if duration < config.min_duration_sec or duration > config.max_duration_sec:
            continue

        path = os.path.join(output_dir, f"sample_{saved:03d}.wav")
        if save_audio(waveform, sr, path, config.target_sample_rate):
            saved += 1

    return saved


def create_synthetic_placeholder(lang_code: str, config: SourceConfig) -> int:
    """Create sine-wave placeholders for languages with no available data.
    These will intentionally produce poor metrics, flagging the data gap."""
    output_dir = os.path.join(config.output_path, lang_code)
    os.makedirs(output_dir, exist_ok=True)

    sr = config.target_sample_rate
    duration = 5.0
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = (0.3 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)

    saved = 0
    for i in range(min(3, config.samples_per_language)):
        path = os.path.join(output_dir, f"placeholder_{i:03d}.wav")
        if save_audio(waveform, sr, path, sr):
            saved += 1
    return saved


def source_indic_audio(config: SourceConfig) -> None:
    """Source test audio for all 22 languages."""
    os.makedirs(config.output_path, exist_ok=True)
    results = {}

    for lang_code, lang_name in tqdm(ALL_LANGUAGES.items(), desc="Sourcing languages"):
        print(f"\n--- {lang_name} ({lang_code}) ---")

        # Check if already sourced
        output_dir = os.path.join(config.output_path, lang_code)
        if os.path.exists(output_dir):
            existing = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
            if len(existing) >= config.samples_per_language:
                print(f"  Already have {len(existing)} samples, skipping")
                results[lang_code] = len(existing)
                continue

        saved = 0

        # Try FLEURS (streaming)
        if lang_code in FLEURS_LANG_MAP:
            print(f"  FLEURS ({FLEURS_LANG_MAP[lang_code]})...")
            saved = source_from_fleurs(lang_code, FLEURS_LANG_MAP[lang_code], config)
            print(f"  Got {saved} samples")

        # Placeholder for languages not in FLEURS
        if saved == 0 and lang_code in NO_SOURCE_LANGS:
            print(f"  No HF source. Creating placeholders.")
            saved = create_synthetic_placeholder(lang_code, config)

        results[lang_code] = saved

    # Summary
    print(f"\n{'='*50}")
    print("SOURCING SUMMARY")
    print(f"{'='*50}")
    total = 0
    for lang_code, count in sorted(results.items()):
        status = "OK" if count >= config.samples_per_language else ("PARTIAL" if count > 0 else "MISSING")
        print(f"  {lang_code} ({ALL_LANGUAGES[lang_code]:>12s}): {count:>3d} [{status}]")
        total += count

    covered = sum(1 for c in results.values() if c > 0)
    real = sum(1 for k, v in results.items() if v > 0 and k not in NO_SOURCE_LANGS)
    print(f"\nTotal: {total} samples, {covered}/22 languages ({real} real + {covered-real} placeholder)")


if __name__ == "__main__":
    config = SourceConfig()
    source_indic_audio(config)
