"""Phase 3 — Per-chunk LID + streaming Whisper fallback.

Two-pass batched architecture:

    Pass 1: load mms-lid-2048, sweep chunks_v2/, write <chunk>.lang.json
            (primary language + confidence)
    Pass 2: cross-check with faster-whisper-large-v3 LID on the same chunks
            agreement → keep, disagreement OR confidence < 0.7 → quarantine

For chunks shorter than 4 seconds, pre/post-pad with ±2s of neighbor audio
from the same source video so LID has enough context to commit. The same
context is reused by the ASR pass (`scripts/retokenize_v2.py --dataset youtube`).

7 Whisper-unsupported Indic languages (brx, doi, kok, mni, sat, snd, kas) get
ASR via `MMSASR` (downstream of this script). LID itself runs on every chunk
through mms-lid-2048 which covers all 22.

Output:
    <chunk>.lang.json next to <chunk>.wav containing
        { language, confidence, method, secondary, agreement, lid_inherits_from_neighbor }

Quarantine: chunks where confidence < LID_CONFIDENCE_THRESHOLD or primary and
secondary disagree are moved to data/chunks_v2_quarantine/ (configurable).

Usage:
    bash autoresearch/scripts/run_safe.sh \\
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/detect_language.py \\
            --chunks_dir data/chunks_v2/yt --rank 0 --world_size 4" 86400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("detect_language")

LID_CONFIDENCE_THRESHOLD = float(os.environ.get("LID_CONFIDENCE_THRESHOLD", 0.7))
STREAMING_LID_THRESHOLD_S = float(os.environ.get("STREAMING_LID_THRESHOLD_S", 4.0))

# ISO mapping: faster-whisper returns ISO 639-1 (2-letter); we normalize to
# ISO 639-3 (3-letter, used by SP tokenizer / sidecar / MMS).
ISO1_TO_ISO3 = {
    "en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel",
    "kn": "kan", "ml": "mal", "mr": "mar", "gu": "guj", "pa": "pan",
    "ur": "urd", "or": "ori", "as": "asm", "ne": "nep", "sa": "san",
}
ISO3_TO_ISO1 = {v: k for k, v in ISO1_TO_ISO3.items()}


def _atomic_write_json(obj: dict, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _normalize_lang(lang: str) -> str:
    if lang in ISO1_TO_ISO3:
        return ISO1_TO_ISO3[lang]
    return lang


class _MMSLid:
    def __init__(self, device: str = "cuda:0"):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-2048")
        model = AutoModelForAudioClassification.from_pretrained(
            "facebook/mms-lid-2048"
        ).to(device)
        model.eval()  # PyTorch inference mode
        self.model = model

    @torch.no_grad()
    def detect(self, audio: torch.Tensor, sample_rate: int) -> tuple[str, float]:
        if sample_rate != 16_000:
            import torchaudio.functional as Faud
            audio = Faud.resample(audio, sample_rate, 16_000)
        if audio.dim() == 2:
            audio = audio[0]
        inputs = self.processor(audio.cpu().numpy(), sampling_rate=16_000,
                                return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top = torch.argmax(probs, dim=-1)
        conf = float(probs[0, top].item())
        label = self.model.config.id2label[int(top)]
        # MMS-lid-2048 returns labels like "hin", "ben" — already ISO 639-3 in most cases.
        return label.lower(), conf


def _load_chunk_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load chunk wav. Returns (audio_array, sr). Uses torchaudio for compatibility."""
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav[0].numpy().astype(np.float32), sr


def _build_streaming_context(
    chunk_path: Path,
    *,
    target_duration_s: float,
    pre_post_seconds: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Concatenate ±pre_post_seconds of neighbor chunks for LID context.

    Neighbors = chunks from the same `source_id` in the parent directory; we
    use chunk_index to order them.
    """
    sidecar = chunk_path.with_suffix(".json")
    if not sidecar.exists():
        return _load_chunk_audio(chunk_path)
    with open(sidecar, encoding="utf-8") as f:
        meta = json.load(f)
    src_id = meta.get("source_id")
    chunk_idx = meta.get("chunk_index")
    parent = chunk_path.parent
    neighbors = []
    for sib in parent.glob("*.json"):
        if sib == sidecar:
            continue
        try:
            with open(sib, encoding="utf-8") as f:
                m = json.load(f)
            if m.get("source_id") == src_id and abs(m.get("chunk_index", -999) - chunk_idx) <= 2:
                neighbors.append((m.get("chunk_index", 0), sib.with_suffix(".wav")))
        except (OSError, json.JSONDecodeError):
            continue
    neighbors.sort()
    audio_arrays = [_load_chunk_audio(chunk_path)[0]]
    sr = 24_000
    pre, post = pre_post_seconds, pre_post_seconds
    for idx, n_path in neighbors:
        if idx < chunk_idx and pre > 0:
            n_arr, n_sr = _load_chunk_audio(n_path)
            slice_n = min(int(pre * n_sr), len(n_arr))
            audio_arrays.insert(0, n_arr[-slice_n:])
            pre -= slice_n / n_sr
            sr = n_sr
        elif idx > chunk_idx and post > 0:
            n_arr, n_sr = _load_chunk_audio(n_path)
            slice_n = min(int(post * n_sr), len(n_arr))
            audio_arrays.append(n_arr[:slice_n])
            post -= slice_n / n_sr
            sr = n_sr
    return np.concatenate(audio_arrays), sr


def detect_chunk_language(
    chunk_path: str,
    primary_lid_model,
    secondary_lid_model=None,
    confidence_threshold: float = LID_CONFIDENCE_THRESHOLD,
    streaming_context_threshold_s: float = STREAMING_LID_THRESHOLD_S,
) -> dict:
    chunk_p = Path(chunk_path)
    sidecar_p = chunk_p.with_suffix(".json")
    duration_s = 0.0
    if sidecar_p.exists():
        try:
            with open(sidecar_p, encoding="utf-8") as f:
                duration_s = float(json.load(f).get("duration_s") or 0.0)
        except (OSError, json.JSONDecodeError):
            pass

    # Streaming context for short clips
    if duration_s and duration_s < streaming_context_threshold_s:
        audio, sr = _build_streaming_context(chunk_p, target_duration_s=duration_s)
        used_streaming = True
    else:
        audio, sr = _load_chunk_audio(chunk_p)
        used_streaming = False

    audio_t = torch.from_numpy(audio)
    primary_lang, primary_conf = primary_lid_model.detect(audio_t, sr)
    primary_lang_iso3 = _normalize_lang(primary_lang)

    secondary_lang = None
    secondary_conf = None
    agreement = None
    if secondary_lid_model is not None:
        try:
            sec_l, sec_c = secondary_lid_model(str(chunk_p))
            secondary_lang = _normalize_lang(sec_l)
            secondary_conf = float(sec_c)
            agreement = (primary_lang_iso3 == secondary_lang)
        except Exception as e:
            log.warning(f"secondary LID failed for {chunk_p.name}: {e}")

    method = "mms_lid_2048"
    if secondary_lang is not None:
        method = "mms_lid_2048+whisper_lid" if agreement else "tie_break"

    return {
        "language": primary_lang_iso3,
        "confidence": primary_conf,
        "method": method,
        "secondary": (
            None if secondary_lang is None
            else {"language": secondary_lang, "confidence": secondary_conf, "method": "whisper_lid"}
        ),
        "agreement": agreement,
        "lid_inherits_from_neighbor": False,
        "used_streaming_context": used_streaming,
        "chunk_duration_s": duration_s,
    }


def _build_secondary_lid(device: str):
    """Returns a callable(path) -> (lang, confidence) using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3",
        device="cuda" if device.startswith("cuda") else "cpu",
        compute_type="float16" if device.startswith("cuda") else "int8",
    )

    def detect(path: str) -> tuple[str, float]:
        _segs, info = model.transcribe(
            path, language=None, beam_size=1, vad_filter=False, no_speech_threshold=0.6,
        )
        return info.language, float(info.language_probability)

    return detect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_dir", required=True)
    p.add_argument("--quarantine_dir", default=None)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true",
                   help="skip chunks that already have a .lang.json")
    p.add_argument("--no_secondary", action="store_true",
                   help="skip Whisper LID (faster, less robust)")
    args = p.parse_args()

    chunks_root = Path(args.chunks_dir)
    quarantine_root = Path(args.quarantine_dir or "data/chunks_v2_quarantine")
    quarantine_root.mkdir(parents=True, exist_ok=True)

    files = sorted(chunks_root.rglob("*.wav"))
    if args.max_files:
        files = files[: args.max_files]
    log.info(f"rank={args.rank}/{args.world_size} chunks={len(files)}")

    primary = _MMSLid(device=args.device)
    secondary = None if args.no_secondary else _build_secondary_lid(args.device)

    decided = 0
    quarantined = 0
    for idx, wav_path in enumerate(files):
        if idx % args.world_size != args.rank:
            continue
        lang_json = wav_path.with_suffix(".lang.json")
        if args.skip_existing and lang_json.exists():
            continue
        try:
            result = detect_chunk_language(
                str(wav_path),
                primary_lid_model=primary,
                secondary_lid_model=secondary,
                confidence_threshold=LID_CONFIDENCE_THRESHOLD,
                streaming_context_threshold_s=STREAMING_LID_THRESHOLD_S,
            )
        except Exception as e:
            log.warning(f"LID failed on {wav_path.name}: {e}")
            continue

        # Quarantine on low confidence or LID disagreement.
        reject = (
            result["confidence"] < LID_CONFIDENCE_THRESHOLD
            or (result["agreement"] is False)
        )
        if reject:
            qdir = quarantine_root / wav_path.parent.name
            qdir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wav_path), str(qdir / wav_path.name))
            chunk_json = wav_path.with_suffix(".json")
            if chunk_json.exists():
                shutil.move(str(chunk_json), str(qdir / chunk_json.name))
            result["quarantine_reason"] = (
                "low_confidence" if result["confidence"] < LID_CONFIDENCE_THRESHOLD
                else "lid_disagreement"
            )
            _atomic_write_json(result, str(qdir / lang_json.name))
            quarantined += 1
        else:
            _atomic_write_json(result, str(lang_json))
            decided += 1

        if (idx + 1) % 200 == 0:
            log.info(f"  rank={args.rank} processed={idx + 1} decided={decided} "
                     f"quarantined={quarantined}")

    log.info(f"done: decided={decided} quarantined={quarantined}")


if __name__ == "__main__":
    main()
