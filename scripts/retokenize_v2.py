"""Phase 1 retokenization driver — FLEURS, IndicVoices, YouTube.

Walks each dataset, calls VoxtralTokenizer.encode (per-call language + task),
writes `data/tokens_v2/{source}/{lang}/{shard}/file.npy` + paired sidecar.

Multi-GPU sharding: launch one process per GPU. Each process consumes a disjoint
slice of files (`(idx % world_size) == rank`). No DDP — independent producers
writing to different shard directories.

Translation pass is lazy (default 10% of files). Background filler (Phase 1.5)
populates the remaining 90% over weeks; Phase 5 evaluation does not block on
full coverage. (plan §6.Phase 1, integration note 3.2.)

Usage:
    # FLEURS (13 langs by default — known per-sample, no Phase 3 LID needed):
    bash autoresearch/scripts/run_safe.sh \
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/retokenize_v2.py \
            --dataset fleurs --languages all --rank 0 --world_size 4" 86400

    # Distribute across 4 GPUs (run each in its own terminal/tmux):
    for r in 0 1 2 3; do
        bash autoresearch/scripts/run_safe.sh \
            "CUDA_VISIBLE_DEVICES=$r uv run scripts/retokenize_v2.py \
                --dataset fleurs --languages all --rank $r --world_size 4 \
                --device cuda:0" 86400 60 90 &
    done

    # IndicVoices (HF_TOKEN required):
    bash autoresearch/scripts/run_safe.sh \
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/retokenize_v2.py \
            --dataset indicvoices --languages hi,ta,bn --rank 0 --world_size 1" 86400

    # YouTube (gated on Phase 3 LID — chunks need <chunk>.lang.json):
    bash autoresearch/scripts/run_safe.sh \
        "CUDA_VISIBLE_DEVICES=0 uv run scripts/retokenize_v2.py \
            --dataset youtube --input_path data/chunks_v2/yt --rank 0 --world_size 1" 86400
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import dotenv
import numpy as np
import torch
import torchaudio.functional as Faud

dotenv.load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("retokenize_v2")


FLEURS_LANGS = {
    "hi": "hi_in", "kn": "kn_in", "ta": "ta_in", "te": "te_in",
    "ml": "ml_in", "bn": "bn_in", "gu": "gu_in", "mr": "mr_in",
    "pa": "pa_in", "ur": "ur_pk", "ne": "ne_np", "or": "or_in",
    "as": "as_in", "en": "en_us",
}
INDICVOICES_LANGS = {
    "hi": "hindi", "kn": "kannada", "ta": "tamil", "te": "telugu",
    "ml": "malayalam", "bn": "bengali", "gu": "gujarati", "mr": "marathi",
    "pa": "punjabi", "or": "odia", "as": "assamese", "ur": "urdu",
}

DEFAULT_OUTPUT = "data/tokens_v2"
CHUNK_SECONDS = 20
MIMI_SR = 24_000


def _maybe_translate(tokenizer, waveform: torch.Tensor, lang: str, prob: float) -> str | None:
    if prob <= 0:
        return None
    if random.random() >= prob:
        return None
    try:
        return tokenizer.translate(waveform, MIMI_SR, language=lang)
    except Exception as e:
        log.warning(f"translate() failed for lang={lang}: {e}")
        return None


def _shard_for_filename(filename: str) -> str:
    return filename[:2]


def _output_paths(output_dir: str, source: str, lang: str, filename: str) -> tuple[Path, Path, Path]:
    """Resolve the output directory and (.npy, .meta.json) paths for a given file."""
    shard = _shard_for_filename(filename)
    out_dir = Path(output_dir) / source / lang / shard
    npy_path = out_dir / f"{filename}.npy"
    meta_path = out_dir / f"{filename}.meta.json"
    return out_dir, npy_path, meta_path


def _already_tokenized(output_dir: str, source: str, lang: str, filename: str) -> bool:
    """Resume-safety: skip files where both the .npy and a v2 sidecar already
    exist on disk. The schema_version check prevents v1 stragglers from
    masquerading as completed v2 work."""
    from voxtral.data.sidecar import SCHEMA_VERSION

    _, npy_path, meta_path = _output_paths(output_dir, source, lang, filename)
    if not (npy_path.exists() and meta_path.exists()):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    return int(meta.get("schema_version", 0)) == SCHEMA_VERSION


def _index_already_tokenized(output_dir: str, source: str, lang: str, idx: int) -> bool:
    """Cheaper resume check that skips BEFORE loading the dataset row.

    The audio hash isn't known until we load the row, so we glob on the
    `<lang>_<idx:08d>_*` prefix. This produces a false-positive only if the
    dataset reorders rows, which neither FLEURS nor IndicVoices do — they
    return rows by stable index.
    """
    from voxtral.data.sidecar import SCHEMA_VERSION

    shard = f"{lang}_{idx:08d}"[:2]
    out_dir = Path(output_dir) / source / lang / shard
    if not out_dir.exists():
        return False
    matches = list(out_dir.glob(f"{lang}_{idx:08d}_*.meta.json"))
    if not matches:
        return False
    try:
        with open(matches[0], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return False
    return int(meta.get("schema_version", 0)) == SCHEMA_VERSION


def _save(
    tokens: torch.Tensor,
    metadata: dict,
    *,
    output_dir: str,
    source: str,
    lang: str,
    filename: str,
    valid_token_mask: list[bool] | None,
    run_id: str,
) -> None:
    from voxtral.data.sidecar import atomic_save_npy, write_metadata_sidecar

    out_dir, npy_path, _meta_path = _output_paths(output_dir, source, lang, filename)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(tokens.detach().cpu().numpy(), str(npy_path))
    write_metadata_sidecar(
        str(npy_path),
        metadata,
        valid_token_mask=valid_token_mask,
        run_id=run_id,
    )


def _make_filename(lang: str, idx: int, audio_hash: str) -> str:
    return f"{lang}_{idx:08d}_{audio_hash[:8]}"


def _hash_audio(arr: np.ndarray) -> str:
    import hashlib
    return hashlib.md5(arr[:1000].tobytes()).hexdigest()


def _resolve_languages(dataset: str, languages: str) -> list[str]:
    pool = FLEURS_LANGS if dataset == "fleurs" else INDICVOICES_LANGS
    if languages == "all":
        return list(pool.keys())
    return [l.strip() for l in languages.split(",") if l.strip()]


def _build_tokenizer(device: str, dtype: torch.dtype):
    from voxtral.tokenizer.model import VoxtralTokenizer, VoxtralTokenizerConfig
    cfg = VoxtralTokenizerConfig()
    tok = VoxtralTokenizer(cfg).to(device=device, dtype=dtype)
    return tok


# ---------------------------------------------------------------------------
# FLEURS
# ---------------------------------------------------------------------------

def _process_fleurs(
    *,
    languages: list[str],
    output_dir: str,
    max_files_per_lang: int | None,
    translation_sample_rate: float,
    rank: int,
    world_size: int,
    device: str,
    dtype: torch.dtype,
):
    from datasets import load_dataset
    from voxtral.data.sidecar import (
        derive_valid_token_mask_from_audio_length,
        new_run_id,
    )

    tokenizer = _build_tokenizer(device, dtype)
    chunk_samples = CHUNK_SECONDS * MIMI_SR
    run_id = new_run_id()
    saved = 0
    skipped = 0
    start = time.time()

    for lang in languages:
        if lang not in FLEURS_LANGS:
            log.warning(f"FLEURS: unknown lang '{lang}', skipping")
            continue
        cfg = FLEURS_LANGS[lang]
        log.info(f"[fleurs:{lang}] loading {cfg}")
        ds = load_dataset("google/fleurs", cfg, split="train", trust_remote_code=True)
        n = min(len(ds), max_files_per_lang) if max_files_per_lang else len(ds)
        resumed = 0
        for idx in range(n):
            if idx % world_size != rank:
                continue
            if _index_already_tokenized(output_dir, "fleurs", lang, idx):
                resumed += 1
                continue
            try:
                row = ds[idx]
                audio = row["audio"]
                arr = audio["array"]
                sr = audio["sampling_rate"]
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                wave = torch.from_numpy(arr.astype(np.float32))
                if sr != MIMI_SR:
                    wave = Faud.resample(wave.unsqueeze(0), sr, MIMI_SR).squeeze(0)
                # legacy 20s chunk path: pad/truncate; valid_token_mask masks the tail
                original_samples = int(wave.shape[0])
                if wave.shape[0] < chunk_samples:
                    wave = torch.nn.functional.pad(wave, (0, chunk_samples - wave.shape[0]))
                else:
                    wave = wave[:chunk_samples]
                wave_in = wave.to(device=device, dtype=dtype).view(1, 1, -1)

                tokens, meta = tokenizer.encode(
                    wave_in, MIMI_SR,
                    language=lang,
                    task="transcribe",
                    source_metadata={
                        "source": "fleurs",
                        "source_id": str(row.get("id") or idx),
                        "data_license_class": "apache",
                        "language_method": "dataset_label",
                        "language_confidence": 1.0,
                    },
                )
                # lazy translate (sampled)
                if random.random() < translation_sample_rate:
                    try:
                        meta["translation_en"] = tokenizer.translate(
                            wave_in, MIMI_SR, language=lang
                        )
                    except Exception as e:
                        log.warning(f"translate failed: {e}")

                tokens_first = tokens[0]
                mask = derive_valid_token_mask_from_audio_length(
                    audio_samples=original_samples,
                    chunk_samples=chunk_samples,
                    token_count=int(tokens_first.numel()),
                    stride=tokenizer.stream_stride,
                )
                fname = _make_filename(lang, idx, _hash_audio(arr))
                _save(
                    tokens_first, meta,
                    output_dir=output_dir, source="fleurs",
                    lang=lang, filename=fname,
                    valid_token_mask=mask, run_id=run_id,
                )
                saved += 1
                if saved % 100 == 0:
                    elapsed = time.time() - start
                    log.info(f"  rank={rank} saved={saved} ({saved/elapsed:.1f}/s)")
            except Exception as e:
                skipped += 1
                log.warning(f"[fleurs:{lang}] idx={idx}: {e}")
        log.info(f"[fleurs:{lang}] done — saved={saved} skipped={skipped} resumed={resumed}")


# ---------------------------------------------------------------------------
# IndicVoices
# ---------------------------------------------------------------------------

def _process_indicvoices(
    *,
    languages: list[str],
    output_dir: str,
    max_files_per_lang: int | None,
    translation_sample_rate: float,
    rank: int,
    world_size: int,
    device: str,
    dtype: torch.dtype,
):
    from datasets import load_dataset
    from voxtral.data.sidecar import (
        derive_valid_token_mask_from_audio_length,
        new_run_id,
    )

    tokenizer = _build_tokenizer(device, dtype)
    chunk_samples = CHUNK_SECONDS * MIMI_SR
    run_id = new_run_id()
    saved = 0
    skipped = 0

    for lang in languages:
        if lang not in INDICVOICES_LANGS:
            log.warning(f"IndicVoices: unknown lang '{lang}', skipping")
            continue
        cfg = INDICVOICES_LANGS[lang]
        log.info(f"[iv:{lang}] streaming {cfg}")
        try:
            ds = load_dataset("ai4bharat/IndicVoices", cfg, split="train", streaming=True)
        except Exception as e:
            log.warning(f"[iv:{lang}] HF load failed: {e}; need HF_TOKEN")
            continue
        idx = 0
        resumed = 0
        for row in ds:
            if idx % world_size != rank:
                idx += 1
                continue
            if max_files_per_lang and saved >= max_files_per_lang * (idx // world_size + 1):
                break
            if _index_already_tokenized(output_dir, "iv", lang, idx):
                resumed += 1
                idx += 1
                continue
            try:
                audio = row["audio_filepath"]
                arr = audio["array"]
                sr = audio["sampling_rate"]
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                wave = torch.from_numpy(arr.astype(np.float32))
                if sr != MIMI_SR:
                    wave = Faud.resample(wave.unsqueeze(0), sr, MIMI_SR).squeeze(0)
                original_samples = int(wave.shape[0])
                if wave.shape[0] < chunk_samples:
                    wave = torch.nn.functional.pad(wave, (0, chunk_samples - wave.shape[0]))
                else:
                    wave = wave[:chunk_samples]
                wave_in = wave.to(device=device, dtype=dtype).view(1, 1, -1)

                tokens, meta = tokenizer.encode(
                    wave_in, MIMI_SR,
                    language=lang,
                    source_metadata={
                        "source": "indicvoices",
                        "source_id": str(row.get("audio_filepath", {}).get("path") or idx),
                        "data_license_class": "cc-by-4.0",
                        "language_method": "dataset_label",
                        "language_confidence": 1.0,
                    },
                )
                if random.random() < translation_sample_rate:
                    try:
                        meta["translation_en"] = tokenizer.translate(
                            wave_in, MIMI_SR, language=lang
                        )
                    except Exception:
                        pass
                tokens_first = tokens[0]
                mask = derive_valid_token_mask_from_audio_length(
                    audio_samples=original_samples,
                    chunk_samples=chunk_samples,
                    token_count=int(tokens_first.numel()),
                    stride=tokenizer.stream_stride,
                )
                fname = _make_filename(lang, idx, _hash_audio(arr))
                _save(
                    tokens_first, meta,
                    output_dir=output_dir, source="iv",
                    lang=lang, filename=fname,
                    valid_token_mask=mask, run_id=run_id,
                )
                saved += 1
            except Exception as e:
                skipped += 1
                if skipped < 10:
                    log.warning(f"[iv:{lang}] idx={idx}: {e}")
            idx += 1
        log.info(f"[iv:{lang}] done — saved={saved} skipped={skipped} resumed={resumed}")


# ---------------------------------------------------------------------------
# YouTube — reads from chunks_v2/yt with paired .lang.json (Phase 3) OR from
# chunks_indic_yt/ with --language fallback (pre-Phase 3 manual override).
# ---------------------------------------------------------------------------

def _process_youtube(
    *,
    input_path: str,
    output_dir: str,
    languages: list[str],
    fallback_lang: str | None,
    max_files: int | None,
    translation_sample_rate: float,
    rank: int,
    world_size: int,
    device: str,
    dtype: torch.dtype,
):
    import torchaudio
    from voxtral.data.sidecar import (
        derive_valid_token_mask_from_audio_length,
        new_run_id,
    )

    tokenizer = _build_tokenizer(device, dtype)
    chunk_samples = CHUNK_SECONDS * MIMI_SR
    run_id = new_run_id()
    saved = 0
    skipped = 0

    src_root = Path(input_path)
    files = []
    for ext in ("*.m4a", "*.wav", "*.flac", "*.webm", "*.mp4"):
        files.extend(src_root.rglob(ext))
    files.sort()
    if max_files:
        files = files[:max_files]

    resumed = 0
    for idx, path in enumerate(files):
        if idx % world_size != rank:
            continue
        try:
            # Phase 3 LID sidecar: <path>.lang.json
            lang_sidecar = path.with_suffix(path.suffix + ".lang.json")
            if lang_sidecar.exists():
                with open(lang_sidecar, encoding="utf-8") as f:
                    lid = json.load(f)
                lang = lid.get("language") or fallback_lang
                lang_method = lid.get("method") or "phase3_lid"
                lang_conf = lid.get("confidence") or 1.0
            else:
                lang = fallback_lang
                lang_method = "fallback_arg"
                lang_conf = 0.5
            if not lang:
                skipped += 1
                continue
            if languages and lang not in languages and "all" not in languages:
                continue

            # Resume-safety: filename is `path.stem` for YouTube, so we can do
            # the full path-based check (no audio_hash variability).
            if _already_tokenized(output_dir, "yt", lang, path.stem):
                resumed += 1
                continue

            wave, sr = torchaudio.load(str(path))
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if sr != MIMI_SR:
                wave = Faud.resample(wave, sr, MIMI_SR)
            original_samples = int(wave.shape[1])
            if wave.shape[1] < chunk_samples:
                wave = torch.nn.functional.pad(wave, (0, chunk_samples - wave.shape[1]))
            else:
                wave = wave[:, :chunk_samples]
            wave_in = wave.to(device=device, dtype=dtype).unsqueeze(0)  # (1, 1, T)

            tokens, meta = tokenizer.encode(
                wave_in, MIMI_SR,
                language=lang,
                source_metadata={
                    "source": "youtube",
                    "source_id": path.stem,
                    "source_url": None,
                    "data_license_class": "research_only",
                    "language_method": lang_method,
                    "language_confidence": float(lang_conf),
                },
            )
            if random.random() < translation_sample_rate:
                try:
                    meta["translation_en"] = tokenizer.translate(
                        wave_in, MIMI_SR, language=lang
                    )
                except Exception:
                    pass

            tokens_first = tokens[0]
            mask = derive_valid_token_mask_from_audio_length(
                audio_samples=original_samples,
                chunk_samples=chunk_samples,
                token_count=int(tokens_first.numel()),
                stride=tokenizer.stream_stride,
            )
            _save(
                tokens_first, meta,
                output_dir=output_dir, source="yt",
                lang=lang, filename=path.stem,
                valid_token_mask=mask, run_id=run_id,
            )
            saved += 1
            if saved % 200 == 0:
                log.info(f"  yt rank={rank} saved={saved} skipped={skipped}")
        except Exception as e:
            skipped += 1
            if skipped < 10:
                log.warning(f"[yt] {path.name}: {e}")

    log.info(f"[yt] done — saved={saved} skipped={skipped} resumed={resumed}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fleurs", "indicvoices", "youtube"])
    p.add_argument("--languages", default="all")
    p.add_argument("--input_path", default=None,
                   help="Source dir for youtube; defaults to data/chunks_v2/yt then chunks_indic_yt")
    p.add_argument("--fallback_lang", default=None,
                   help="YouTube only: language to assume when no .lang.json sidecar exists")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--max_files_per_lang", type=int, default=None)
    p.add_argument("--max_files", type=int, default=None, help="YouTube only")
    p.add_argument("--translation_sample_rate", type=float,
                   default=float(os.environ.get("TRANSLATION_SAMPLE_RATE", 0.10)))
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args()

    random.seed(args.seed + args.rank)  # stagger so different ranks pick different translate samples
    torch.manual_seed(args.seed + args.rank)

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    languages = _resolve_languages(args.dataset, args.languages) if args.dataset != "youtube" else (
        [l.strip() for l in args.languages.split(",") if l.strip()]
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"rank={args.rank}/{args.world_size} dataset={args.dataset} "
             f"langs={languages} output={args.output_dir}")

    if args.dataset == "fleurs":
        _process_fleurs(
            languages=languages, output_dir=args.output_dir,
            max_files_per_lang=args.max_files_per_lang,
            translation_sample_rate=args.translation_sample_rate,
            rank=args.rank, world_size=args.world_size,
            device=args.device, dtype=dtype,
        )
    elif args.dataset == "indicvoices":
        _process_indicvoices(
            languages=languages, output_dir=args.output_dir,
            max_files_per_lang=args.max_files_per_lang,
            translation_sample_rate=args.translation_sample_rate,
            rank=args.rank, world_size=args.world_size,
            device=args.device, dtype=dtype,
        )
    elif args.dataset == "youtube":
        input_path = args.input_path
        if input_path is None:
            cand = REPO_ROOT / "data" / "chunks_v2" / "yt"
            input_path = str(cand if cand.exists() else REPO_ROOT / "data" / "chunks_indic_yt")
        _process_youtube(
            input_path=input_path, output_dir=args.output_dir,
            languages=languages, fallback_lang=args.fallback_lang,
            max_files=args.max_files,
            translation_sample_rate=args.translation_sample_rate,
            rank=args.rank, world_size=args.world_size,
            device=args.device, dtype=dtype,
        )
    else:
        log.error(f"unknown dataset {args.dataset}")
        sys.exit(2)


if __name__ == "__main__":
    main()
