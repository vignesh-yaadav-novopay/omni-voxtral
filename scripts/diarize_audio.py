"""Diarization pipeline for dual-stream training data.

Takes single-channel conversational audio and produces dual-stream token sequences
by separating speakers using pyannote.audio diarization.

Pipeline:
1. Load audio file
2. Run speaker diarization → per-speaker timestamps
3. Extract per-speaker audio tracks (zero out other speaker)
4. Encode both tracks with DualStreamTokenizer
5. Save as .npy for training

Usage:
    # Process a single file:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/diarize_audio.py --input audio.wav

    # Process a directory:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/diarize_audio.py --input_dir data/chunks/ --output_dir data/dual_tokens/

    # Dry run (diarize only, don't tokenize):
    uv run scripts/diarize_audio.py --input audio.wav --diarize_only

Requires:
    pip install pyannote.audio
    Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
    HF_TOKEN must be set in .env
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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

# Target sample rate for Mimi codec
MIMI_SR = 24_000


def load_audio(path: str, target_sr: int = MIMI_SR) -> torch.Tensor:
    """Load and resample audio to mono at target sample rate.

    Returns: (1, samples) tensor
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def diarize(audio_path: str, device: str = "cuda:0") -> list[dict]:
    """Run speaker diarization on an audio file.

    Returns list of segments: [{"speaker": "SPEAKER_00", "start": 0.5, "end": 2.3}, ...]
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.error(
            "pyannote.audio not installed. Install with: uv pip install pyannote.audio\n"
            "Then accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN not set. Required for pyannote model access.")
        sys.exit(1)

    logger.info("Loading pyannote speaker-diarization-3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    pipeline = pipeline.to(torch.device(device))

    logger.info(f"Diarizing {audio_path}...")
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })

    # Sort by start time
    segments.sort(key=lambda s: s["start"])

    # Log summary
    speakers = set(s["speaker"] for s in segments)
    logger.info(
        f"Found {len(speakers)} speakers, {len(segments)} segments, "
        f"total {segments[-1]['end']:.1f}s" if segments else "no segments"
    )
    for spk in sorted(speakers):
        spk_dur = sum(s["end"] - s["start"] for s in segments if s["speaker"] == spk)
        logger.info(f"  {spk}: {spk_dur:.1f}s")

    return segments


def separate_speakers(
    waveform: torch.Tensor,
    segments: list[dict],
    sr: int = MIMI_SR,
) -> dict[str, torch.Tensor]:
    """Separate audio into per-speaker tracks using diarization segments.

    For each speaker, creates a track where other speakers' regions are zeroed out.
    This is a simple "masking" approach — no source separation model needed.

    Args:
        waveform: (1, samples) mono audio
        segments: diarization output
        sr: sample rate

    Returns:
        dict mapping speaker ID → (1, samples) tensor
    """
    speakers = sorted(set(s["speaker"] for s in segments))
    total_samples = waveform.shape[1]
    tracks = {}

    for speaker in speakers:
        track = torch.zeros_like(waveform)
        for seg in segments:
            if seg["speaker"] == speaker:
                start_sample = int(seg["start"] * sr)
                end_sample = min(int(seg["end"] * sr), total_samples)
                track[:, start_sample:end_sample] = waveform[:, start_sample:end_sample]
        tracks[speaker] = track

    return tracks


def assign_roles(
    tracks: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign speaker tracks to user/model roles.

    Heuristic: the speaker with more total speech is the "model" (interviewer/host),
    the other is the "user" (guest/caller). For training, the assignment is arbitrary
    since the model learns both roles.

    Returns:
        (user_audio, model_audio) each (1, samples)
    """
    speakers = sorted(tracks.keys())

    if len(speakers) == 1:
        # Single speaker: duplicate as both streams (silence in one)
        logger.warning("Only 1 speaker found. Using same audio for both streams.")
        return tracks[speakers[0]], tracks[speakers[0]]

    if len(speakers) > 2:
        # More than 2 speakers: take the two with most speech
        speech_durations = {
            spk: (track.abs() > 0.01).sum().item()
            for spk, track in tracks.items()
        }
        top_two = sorted(speech_durations, key=speech_durations.get, reverse=True)[:2]
        logger.info(f"Multiple speakers found, using top 2: {top_two}")
        speakers = top_two

    # Assign: more speech = model (host), less = user (guest)
    dur_0 = (tracks[speakers[0]].abs() > 0.01).sum().item()
    dur_1 = (tracks[speakers[1]].abs() > 0.01).sum().item()

    if dur_0 >= dur_1:
        model_audio, user_audio = tracks[speakers[0]], tracks[speakers[1]]
    else:
        model_audio, user_audio = tracks[speakers[1]], tracks[speakers[0]]

    return user_audio, model_audio


def process_file(
    audio_path: str,
    output_path: str | None = None,
    device: str = "cuda:0",
    diarize_only: bool = False,
) -> dict:
    """Process a single audio file through the full diarization pipeline.

    Returns dict with statistics.
    """
    logger.info(f"Processing: {audio_path}")

    # Step 1: Diarize
    segments = diarize(audio_path, device=device)
    if not segments:
        logger.warning(f"No speech segments found in {audio_path}")
        return {"status": "no_speech", "path": audio_path}

    if diarize_only:
        return {
            "status": "diarized",
            "path": audio_path,
            "num_speakers": len(set(s["speaker"] for s in segments)),
            "num_segments": len(segments),
            "duration": segments[-1]["end"],
        }

    # Step 2: Load audio and separate speakers
    waveform = load_audio(audio_path, target_sr=MIMI_SR)
    tracks = separate_speakers(waveform, segments, sr=MIMI_SR)
    user_audio, model_audio = assign_roles(tracks)

    # Step 3: Ensure same length, add batch dim for tokenizer
    min_len = min(user_audio.shape[1], model_audio.shape[1])
    user_audio = user_audio[:, :min_len].unsqueeze(0)   # (1, 1, samples)
    model_audio = model_audio[:, :min_len].unsqueeze(0)  # (1, 1, samples)

    # Step 4: Encode with DualStreamTokenizer
    from voxtral.tokenizer.dual_stream import DualStreamTokenizer
    from voxtral.tokenizer.model import VoxtralTokenizerConfig

    tokenizer_config = VoxtralTokenizerConfig()
    tokenizer = DualStreamTokenizer(tokenizer_config)
    tokenizer = tokenizer.to(device)

    tokens = tokenizer.encode(user_audio, model_audio, sample_rate=MIMI_SR)
    tokens_np = tokens.cpu().numpy()

    # Step 5: Save
    if output_path is None:
        output_path = audio_path.replace(".wav", "_dual.npy").replace(".mp3", "_dual.npy")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, tokens_np)

    logger.info(f"Saved {tokens_np.shape} tokens to {output_path}")

    return {
        "status": "success",
        "path": audio_path,
        "output": output_path,
        "token_shape": tokens_np.shape,
        "duration_sec": min_len / MIMI_SR,
        "num_speakers": len(set(s["speaker"] for s in segments)),
    }


def main():
    parser = argparse.ArgumentParser(description="Diarize audio for dual-stream training")
    parser.add_argument("--input", type=str, help="Single audio file to process")
    parser.add_argument("--input_dir", type=str, help="Directory of audio files")
    parser.add_argument("--output_dir", type=str, default="./data/dual_tokens", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for models")
    parser.add_argument("--diarize_only", action="store_true", help="Only diarize, don't tokenize")
    parser.add_argument("--extensions", type=str, default=".wav,.mp3,.flac,.ogg", help="Audio extensions")
    args = parser.parse_args()

    if args.input:
        process_file(args.input, device=args.device, diarize_only=args.diarize_only)
    elif args.input_dir:
        extensions = tuple(args.extensions.split(","))
        audio_files = []
        for root, _, files in os.walk(args.input_dir):
            for f in files:
                if f.endswith(extensions):
                    audio_files.append(os.path.join(root, f))

        logger.info(f"Found {len(audio_files)} audio files in {args.input_dir}")

        results = []
        for audio_file in audio_files:
            rel_path = os.path.relpath(audio_file, args.input_dir)
            output_path = os.path.join(
                args.output_dir,
                os.path.splitext(rel_path)[0] + ".npy",
            )
            try:
                result = process_file(
                    audio_file,
                    output_path=output_path,
                    device=args.device,
                    diarize_only=args.diarize_only,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                results.append({"status": "error", "path": audio_file, "error": str(e)})

        # Summary
        success = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Processed {len(results)} files: {success} success, {len(results) - success} failed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
