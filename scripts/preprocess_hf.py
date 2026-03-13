"""Preprocess audio from HuggingFace datasets directly to training tokens.

Streams audio from FLEURS/IndicVoices → tokenizes with VoxtralTokenizer on GPU →
saves as .npy files. No intermediate wav files needed.

Usage:
    # Process FLEURS (13 Indic languages, fully open):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/preprocess_hf.py --dataset fleurs --max_samples 1000

    # Process IndicVoices (22 languages, needs HF_TOKEN):
    CUDA_VISIBLE_DEVICES=0 uv run scripts/preprocess_hf.py --dataset indicvoices --max_samples 5000

    # Process specific languages only:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/preprocess_hf.py --dataset fleurs --languages hi,kn,ta

    # Dry run (count samples only):
    uv run scripts/preprocess_hf.py --dataset fleurs --dry_run
"""

import argparse
import hashlib
import logging
import os
import time

import dotenv
import numpy as np
import torch
import torchaudio

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# FLEURS language configs (ISO code → FLEURS config name)
FLEURS_LANGS = {
    "hi": "hi_in", "kn": "kn_in", "ta": "ta_in", "te": "te_in",
    "ml": "ml_in", "bn": "bn_in", "gu": "gu_in", "mr": "mr_in",
    "pa": "pa_in", "ur": "ur_pk", "ne": "ne_np", "or": "or_in",
    "as": "as_in", "en": "en_us",
}

# IndicVoices language configs
INDICVOICES_LANGS = {
    "hi": "hindi", "kn": "kannada", "ta": "tamil", "te": "telugu",
    "ml": "malayalam", "bn": "bengali", "gu": "gujarati", "mr": "marathi",
    "pa": "punjabi", "ur": "urdu", "ne": "nepali", "or": "odia",
    "as": "assamese", "brx": "bodo", "doi": "dogri", "kok": "konkani",
    "mai": "maithili", "mni": "manipuri", "sa": "sanskrit",
    "sat": "santali", "sd": "sindhi", "ks": "kashmiri",
}

MIMI_SR = 24_000
CHUNK_SECONDS = 20  # Match existing preprocessing


def get_dataset_iterator(dataset_name: str, language: str, split: str = "train"):
    """Get a streaming iterator for a HuggingFace audio dataset.

    Returns iterator of dicts with 'audio' key containing {'array': np.ndarray, 'sampling_rate': int}.
    """
    from datasets import load_dataset

    if dataset_name == "fleurs":
        config = FLEURS_LANGS.get(language)
        if not config:
            logger.warning(f"Language {language} not in FLEURS")
            return None
        ds = load_dataset("google/fleurs", config, split=split, streaming=True, trust_remote_code=True)
        return ds

    elif dataset_name == "indicvoices":
        config = INDICVOICES_LANGS.get(language)
        if not config:
            logger.warning(f"Language {language} not in IndicVoices")
            return None
        ds = load_dataset("ai4bharat/IndicVoices", config, split=split, streaming=True, trust_remote_code=True)
        return ds

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def extract_audio(sample: dict, dataset_name: str) -> tuple[np.ndarray, int] | None:
    """Extract audio array and sample rate from a dataset sample."""
    try:
        if dataset_name == "fleurs":
            audio = sample["audio"]
            return audio["array"], audio["sampling_rate"]
        elif dataset_name == "indicvoices":
            audio = sample["audio"]
            return audio["array"], audio["sampling_rate"]
        return None
    except Exception as e:
        logger.debug(f"Failed to extract audio: {e}")
        return None


def preprocess_sample(
    audio_array: np.ndarray,
    sr: int,
    tokenizer: torch.nn.Module,
    device: torch.device,
    chunk_samples: int,
) -> torch.Tensor | None:
    """Tokenize a single audio sample.

    Resamples to 24kHz, pads/truncates to chunk_samples, then encodes.
    Returns token tensor or None on failure.
    """
    try:
        waveform = torch.from_numpy(audio_array).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != MIMI_SR:
            waveform = torchaudio.functional.resample(waveform, sr, MIMI_SR)

        # Pad or truncate
        if waveform.shape[1] < chunk_samples:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_samples - waveform.shape[1]))
        elif waveform.shape[1] > chunk_samples:
            waveform = waveform[:, :chunk_samples]

        # Add batch dim: (1, 1, samples) — keep on CPU
        # VoxtralTokenizer.encode() handles device transfer:
        #   - Whisper needs CPU input (.numpy())
        #   - Mimi moves to self.device internally
        # Cast to match tokenizer dtype (fp16)
        waveform = waveform.unsqueeze(0).half()

        tokens = tokenizer.encode(waveform, MIMI_SR)
        return tokens.cpu()

    except Exception as e:
        logger.warning(f"Tokenization failed: {e}")
        return None


def make_filename(language: str, index: int, audio_hash: str) -> str:
    """Create a deterministic filename for a token file."""
    return f"{language}_{index:06d}_{audio_hash[:8]}"


def preprocess_dataset(
    dataset_name: str,
    languages: list[str],
    output_dir: str,
    max_samples_per_lang: int | None = None,
    device: str = "cuda:0",
    dry_run: bool = False,
):
    """Process a HuggingFace dataset into training tokens."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_samples = CHUNK_SECONDS * MIMI_SR

    # Initialize tokenizer (only if not dry run)
    tokenizer = None
    if not dry_run:
        from voxtral.tokenizer.model import VoxtralTokenizer, VoxtralTokenizerConfig

        logger.info("Loading tokenizer (Mimi + Whisper)...")
        tok_config = VoxtralTokenizerConfig()
        tokenizer = VoxtralTokenizer(tok_config)
        tokenizer = tokenizer.to(device=torch.device(device), dtype=torch.float16)
        logger.info("Tokenizer ready.")

    total_saved = 0
    total_skipped = 0
    start_time = time.time()

    for lang in languages:
        logger.info(f"Processing {dataset_name}/{lang}...")
        ds = get_dataset_iterator(dataset_name, lang)
        if ds is None:
            continue

        lang_saved = 0
        lang_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # Count existing files to skip already-processed samples
        existing_count = len([f for f in os.listdir(lang_dir) if os.path.isdir(os.path.join(lang_dir, f))
                              for ff in os.listdir(os.path.join(lang_dir, f)) if ff.endswith('.npy')])
        if existing_count > 0:
            logger.info(f"  {lang}: {existing_count} already exist, continuing from there")
            lang_saved = existing_count

        for i, sample in enumerate(ds):
            if max_samples_per_lang and lang_saved >= max_samples_per_lang:
                break

            result = extract_audio(sample, dataset_name)
            if result is None:
                total_skipped += 1
                continue

            audio_array, sr = result

            # Skip very short audio (< 1 second)
            duration_sec = len(audio_array) / sr
            if duration_sec < 1.0:
                total_skipped += 1
                continue

            if dry_run:
                lang_saved += 1
                total_saved += 1
                if lang_saved % 100 == 0:
                    logger.info(f"  {lang}: {lang_saved} samples counted ({duration_sec:.1f}s)")
                continue

            # Skip samples that were already processed in a previous run
            if i < existing_count:
                continue

            # Tokenize
            tokens = preprocess_sample(audio_array, sr, tokenizer, torch.device(device), chunk_samples)
            if tokens is None:
                total_skipped += 1
                continue

            # Save
            audio_hash = hashlib.md5(audio_array[:1000].tobytes()).hexdigest()
            filename = make_filename(lang, lang_saved, audio_hash)
            subdir = filename[:2]
            save_dir = os.path.join(lang_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{filename}.npy"), tokens.numpy())

            lang_saved += 1
            total_saved += 1

            if lang_saved % 50 == 0:
                elapsed = time.time() - start_time
                rate = total_saved / elapsed
                logger.info(
                    f"  {lang}: {lang_saved} saved | "
                    f"total: {total_saved} ({rate:.1f} samples/sec)"
                )

        logger.info(f"  {lang}: {lang_saved} samples {'counted' if dry_run else 'saved'}")

    elapsed = time.time() - start_time
    logger.info(f"\nDone! {total_saved} saved, {total_skipped} skipped in {elapsed:.0f}s")
    logger.info(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess HF datasets to training tokens")
    parser.add_argument("--dataset", type=str, required=True, choices=["fleurs", "indicvoices"])
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language codes (default: all)")
    parser.add_argument("--output_dir", type=str, default="./data/tokens", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per language")
    parser.add_argument("--device", type=str, default="cuda:0", help="GPU device")
    parser.add_argument("--dry_run", action="store_true", help="Count samples only")
    args = parser.parse_args()

    if args.languages:
        languages = args.languages.split(",")
    else:
        if args.dataset == "fleurs":
            languages = list(FLEURS_LANGS.keys())
        else:
            languages = list(INDICVOICES_LANGS.keys())

    preprocess_dataset(
        dataset_name=args.dataset,
        languages=languages,
        output_dir=args.output_dir,
        max_samples_per_lang=args.max_samples,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
