"""Benchmark Mimi codec on 22 Indic languages.

Encodes audio → Mimi tokens → decodes back, measures quality metrics.
Tests both 4 and 8 quantizer configurations.

Usage: CUDA_VISIBLE_DEVICES=0 uv run scripts/benchmark_codec.py

GPU requirement: ~4GB VRAM (Mimi ~300MB + Whisper-large-v3 ~3GB)
"""

import json
import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from pesq import pesq
from pydantic_settings import BaseSettings
from tqdm import tqdm

from voxtral.tokenizer.mimi.models.loaders import DEFAULT_REPO, MIMI_NAME, get_mimi

# Language metadata
ALL_LANGUAGES = {
    "asm": "Assamese", "ben": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "guj": "Gujarati", "hin": "Hindi", "kan": "Kannada", "kas": "Kashmiri",
    "kok": "Konkani", "mai": "Maithili", "mal": "Malayalam", "mni": "Manipuri",
    "mar": "Marathi", "nep": "Nepali", "ori": "Odia", "pan": "Punjabi",
    "san": "Sanskrit", "sat": "Santali", "snd": "Sindhi", "tam": "Tamil",
    "tel": "Telugu", "urd": "Urdu",
}


class CodecBenchmarkConfig(BaseSettings):
    input_path: str = "./data/indic_test_samples"
    output_path: str = "./data/codec_benchmark_results"
    num_quantizers: list[int] = [4, 8]
    mimi_sample_rate: int = 24000
    pesq_sample_rate: int = 16000  # PESQ requires 8kHz or 16kHz
    device: str = "cuda:0"


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load audio file, resample to target_sr, return (1, 1, T) tensor."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Shape: (1, 1, T) for Mimi
    return waveform.unsqueeze(0)


def compute_pesq_score(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> float:
    """Compute PESQ score. Requires 8kHz or 16kHz."""
    try:
        if sr not in (8000, 16000):
            raise ValueError(f"PESQ requires 8kHz or 16kHz, got {sr}")
        mode = "wb" if sr == 16000 else "nb"
        score = pesq(sr, original, reconstructed, mode)
        return float(score)
    except Exception as e:
        return float("nan")


def compute_si_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Noise Ratio (dB)."""
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    ref = original[:min_len]
    est = reconstructed[:min_len]

    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    dot = np.dot(ref, est)
    s_target = dot * ref / (np.dot(ref, ref) + 1e-8)
    e_noise = est - s_target

    si_snr = 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8) + 1e-8)
    return float(si_snr)


def compute_f0_correlation(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> float:
    """F0 (pitch) correlation using pyin."""
    try:
        f0_orig, _, _ = librosa.pyin(original, fmin=50, fmax=500, sr=sr)
        f0_recon, _, _ = librosa.pyin(reconstructed, fmin=50, fmax=500, sr=sr)

        # Remove NaN frames (unvoiced)
        min_len = min(len(f0_orig), len(f0_recon))
        f0_orig = f0_orig[:min_len]
        f0_recon = f0_recon[:min_len]
        mask = ~(np.isnan(f0_orig) | np.isnan(f0_recon))

        if mask.sum() < 10:  # Too few voiced frames
            return float("nan")

        corr = np.corrcoef(f0_orig[mask], f0_recon[mask])[0, 1]
        return float(corr)
    except Exception:
        return float("nan")


def compute_energy_correlation(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> float:
    """RMS energy envelope correlation."""
    try:
        rms_orig = librosa.feature.rms(y=original, frame_length=2048, hop_length=512)[0]
        rms_recon = librosa.feature.rms(y=reconstructed, frame_length=2048, hop_length=512)[0]

        min_len = min(len(rms_orig), len(rms_recon))
        rms_orig = rms_orig[:min_len]
        rms_recon = rms_recon[:min_len]

        if min_len < 5:
            return float("nan")

        corr = np.corrcoef(rms_orig, rms_recon)[0, 1]
        return float(corr)
    except Exception:
        return float("nan")


def compute_speaking_rate_ratio(original: np.ndarray, reconstructed: np.ndarray, sr: int) -> float:
    """Ratio of voiced frames (proxy for speaking rate preservation)."""
    try:
        _, voiced_orig, _ = librosa.pyin(original, fmin=50, fmax=500, sr=sr)
        _, voiced_recon, _ = librosa.pyin(reconstructed, fmin=50, fmax=500, sr=sr)

        min_len = min(len(voiced_orig), len(voiced_recon))
        rate_orig = np.nanmean(voiced_orig[:min_len])
        rate_recon = np.nanmean(voiced_recon[:min_len])

        if rate_orig < 1e-6:
            return float("nan")

        return float(rate_recon / (rate_orig + 1e-8))
    except Exception:
        return float("nan")


@torch.no_grad()
def benchmark_single(
    audio_path: str,
    mimi: torch.nn.Module,
    num_q: int,
    config: CodecBenchmarkConfig,
) -> dict:
    """Run roundtrip encode→decode on a single audio file, compute metrics."""
    device = torch.device(config.device)

    # Load audio at Mimi's sample rate (24kHz)
    waveform = load_audio(audio_path, config.mimi_sample_rate).to(device)
    duration_sec = waveform.shape[-1] / config.mimi_sample_rate

    # Set quantizer count
    mimi.set_num_codebooks(num_q)

    # Encode
    t_enc_start = time.perf_counter()
    codes = mimi.encode(waveform)
    torch.cuda.synchronize()
    t_enc_end = time.perf_counter()
    encode_ms = (t_enc_end - t_enc_start) * 1000

    # Decode
    t_dec_start = time.perf_counter()
    reconstructed = mimi.decode(codes)
    torch.cuda.synchronize()
    t_dec_end = time.perf_counter()
    decode_ms = (t_dec_end - t_dec_start) * 1000

    # To numpy for metrics
    orig_24k = waveform.squeeze().cpu().numpy()
    recon_24k = reconstructed.squeeze().cpu().numpy()

    # Ensure same length
    min_len = min(len(orig_24k), len(recon_24k))
    orig_24k = orig_24k[:min_len]
    recon_24k = recon_24k[:min_len]

    # Resample to 16kHz for PESQ
    orig_16k = librosa.resample(orig_24k, orig_sr=24000, target_sr=16000)
    recon_16k = librosa.resample(recon_24k, orig_sr=24000, target_sr=16000)

    return {
        "file": os.path.basename(audio_path),
        "duration_sec": round(duration_sec, 2),
        "num_quantizers": num_q,
        "pesq": compute_pesq_score(orig_16k, recon_16k, 16000),
        "si_snr_db": compute_si_snr(orig_24k, recon_24k),
        "f0_correlation": compute_f0_correlation(orig_24k, recon_24k, 24000),
        "energy_correlation": compute_energy_correlation(orig_24k, recon_24k, 24000),
        "speaking_rate_ratio": compute_speaking_rate_ratio(orig_24k, recon_24k, 24000),
        "encode_ms": round(encode_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "encode_rtf": round(encode_ms / (duration_sec * 1000), 4),
        "decode_rtf": round(decode_ms / (duration_sec * 1000), 4),
    }


def classify_metric(value: float, pass_thresh: float, marginal_thresh: float, higher_is_better: bool = True) -> str:
    """Classify a metric as PASS, MARGINAL, or FAIL."""
    if np.isnan(value):
        return "N/A"
    if higher_is_better:
        if value >= pass_thresh:
            return "PASS"
        elif value >= marginal_thresh:
            return "MARGINAL"
        else:
            return "FAIL"
    else:
        if value <= pass_thresh:
            return "PASS"
        elif value <= marginal_thresh:
            return "MARGINAL"
        else:
            return "FAIL"


def generate_summary(df: pd.DataFrame, output_path: str) -> str:
    """Generate markdown summary of benchmark results."""
    lines = ["# Mimi Codec Benchmark Results — Indic Languages\n"]
    lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")

    for num_q in sorted(df["num_quantizers"].unique()):
        subset = df[df["num_quantizers"] == num_q]
        lines.append(f"\n## {num_q} Quantizers\n")

        # Per-language aggregation
        lang_stats = subset.groupby("language").agg({
            "pesq": "mean",
            "si_snr_db": "mean",
            "f0_correlation": "mean",
            "energy_correlation": "mean",
            "speaking_rate_ratio": "mean",
            "encode_ms": "mean",
            "decode_ms": "mean",
            "file": "count",
        }).rename(columns={"file": "n_samples"})

        lines.append("| Language | Samples | PESQ | SI-SNR (dB) | F0 Corr | Energy Corr | Spk Rate | Enc (ms) | Dec (ms) | Verdict |")
        lines.append("|----------|---------|------|-------------|---------|-------------|----------|----------|----------|---------|")

        pass_count = 0
        for lang_code, row in lang_stats.iterrows():
            pesq_status = classify_metric(row["pesq"], 3.0, 2.5)
            si_snr_status = classify_metric(row["si_snr_db"], 15, 10)
            f0_status = classify_metric(row["f0_correlation"], 0.9, 0.8)

            # Overall verdict: PASS if PESQ and at least 2 others pass
            statuses = [pesq_status, si_snr_status, f0_status]
            pass_metrics = sum(1 for s in statuses if s == "PASS")
            fail_metrics = sum(1 for s in statuses if s == "FAIL")

            if pass_metrics >= 2 and pesq_status != "FAIL":
                verdict = "PASS"
                pass_count += 1
            elif fail_metrics >= 2:
                verdict = "FAIL"
            else:
                verdict = "MARGINAL"

            lang_name = ALL_LANGUAGES.get(lang_code, lang_code)
            lines.append(
                f"| {lang_name} ({lang_code}) | {int(row['n_samples'])} | "
                f"{row['pesq']:.2f} | {row['si_snr_db']:.1f} | "
                f"{row['f0_correlation']:.3f} | {row['energy_correlation']:.3f} | "
                f"{row['speaking_rate_ratio']:.3f} | "
                f"{row['encode_ms']:.0f} | {row['decode_ms']:.0f} | "
                f"**{verdict}** |"
            )

        total_langs = len(lang_stats)
        pass_pct = (pass_count / total_langs * 100) if total_langs > 0 else 0
        lines.append(f"\n**Summary: {pass_count}/{total_langs} languages PASS ({pass_pct:.0f}%)**\n")

        # Decision recommendation
        if pass_pct >= 80:
            lines.append("> **RECOMMENDATION:** Mimi is viable at this quantizer setting. Proceed to Phase 2.\n")
        elif pass_pct >= 50:
            lines.append("> **RECOMMENDATION:** Fine-tune Mimi on IndicVoices for failing languages.\n")
        else:
            lines.append("> **RECOMMENDATION:** Mimi may not be suitable. Evaluate DM-Codec or train new codec.\n")

    # 4q vs 8q comparison
    if len(df["num_quantizers"].unique()) > 1:
        lines.append("\n## 4q vs 8q Comparison\n")
        for lang_code in sorted(df["language"].unique()):
            lang_data = df[df["language"] == lang_code]
            q4 = lang_data[lang_data["num_quantizers"] == 4]["pesq"].mean()
            q8 = lang_data[lang_data["num_quantizers"] == 8]["pesq"].mean()
            diff = q8 - q4
            lang_name = ALL_LANGUAGES.get(lang_code, lang_code)
            lines.append(f"- **{lang_name}**: 4q={q4:.2f}, 8q={q8:.2f}, delta={diff:+.2f}")

    summary_text = "\n".join(lines)

    summary_path = os.path.join(output_path, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)

    return summary_text


def run_benchmark(config: CodecBenchmarkConfig) -> None:
    """Main benchmark entry point."""
    os.makedirs(config.output_path, exist_ok=True)
    device = torch.device(config.device)

    # Load Mimi
    print(f"Loading Mimi codec on {config.device}...")
    import huggingface_hub as hf_hub
    mimi_weight = hf_hub.hf_hub_download(DEFAULT_REPO, MIMI_NAME)
    mimi = get_mimi(mimi_weight, device=config.device)

    # Discover test samples
    if not os.path.exists(config.input_path):
        print(f"ERROR: Input path {config.input_path} does not exist.")
        print("Run `uv run scripts/source_indic_audio.py` first.")
        return

    lang_dirs = sorted([
        d for d in os.listdir(config.input_path)
        if os.path.isdir(os.path.join(config.input_path, d))
    ])

    if not lang_dirs:
        print(f"ERROR: No language directories found in {config.input_path}")
        return

    print(f"Found {len(lang_dirs)} languages: {', '.join(lang_dirs)}")

    all_results = []

    for lang_code in tqdm(lang_dirs, desc="Languages"):
        lang_dir = os.path.join(config.input_path, lang_code)
        audio_files = sorted([
            f for f in os.listdir(lang_dir)
            if f.endswith(".wav")
        ])

        if not audio_files:
            print(f"  {lang_code}: No .wav files found, skipping")
            continue

        for num_q in config.num_quantizers:
            for audio_file in audio_files:
                audio_path = os.path.join(lang_dir, audio_file)
                try:
                    result = benchmark_single(audio_path, mimi, num_q, config)
                    result["language"] = lang_code
                    all_results.append(result)
                except Exception as e:
                    print(f"  ERROR on {lang_code}/{audio_file} (q={num_q}): {e}")
                    all_results.append({
                        "language": lang_code,
                        "file": audio_file,
                        "num_quantizers": num_q,
                        "pesq": float("nan"),
                        "si_snr_db": float("nan"),
                        "f0_correlation": float("nan"),
                        "energy_correlation": float("nan"),
                        "speaking_rate_ratio": float("nan"),
                        "encode_ms": float("nan"),
                        "decode_ms": float("nan"),
                        "encode_rtf": float("nan"),
                        "decode_rtf": float("nan"),
                        "duration_sec": 0,
                        "error": str(e),
                    })

    # Save raw results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(config.output_path, "raw_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved to {csv_path}")

    # Save as JSON too
    json_path = os.path.join(config.output_path, "raw_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(df, config.output_path)
    print("\n" + summary)


if __name__ == "__main__":
    config = CodecBenchmarkConfig()
    run_benchmark(config)
