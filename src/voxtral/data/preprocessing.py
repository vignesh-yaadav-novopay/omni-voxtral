"""v1 preprocessing entrypoint, adapted for Phase 1.

The legacy 20-second hard-chunk path is preserved (callers that still use it)
but the new contracts are:
- `tokenizer.encode(...)` returns `(tokens, metadata)` — `_save_tokens_v2` writes
  both the .npy and the .meta.json atomically (`<path>.tmp` → `os.replace`).
- `AudioChunkDataset.chunk_frames` is optional. When `None`, audio is returned
  unpadded/uncut for variable-length VAD-chunked inputs (Phase 2 onward).
- A `valid_token_mask` is derived from the original audio length whenever the
  legacy padding path is used so Phase 1.5 trainer can ignore the silent tail.

Phase 4 conversational driver lives in `scripts/diarize_v2.py` — this module is
the single-stream legacy path only.
"""

import os
import typing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydantic_settings as pyds
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voxtral.data.sidecar import (
    atomic_save_npy,
    derive_valid_token_mask_from_audio_length,
    new_run_id,
    write_metadata_sidecar,
)
from voxtral.tokenizer.model import VoxtralTokenizer, VoxtralTokenizerConfig


class PreprocessingConfig(pyds.BaseSettings):
    voxtral_tokenizer_config: VoxtralTokenizerConfig = VoxtralTokenizerConfig()
    input_path: str = "./data/chunks"
    output_path: str = "./data/tokens_v2/legacy"
    language: str = "en"  # MUST be set by caller; v2 callers should override per-file
    source: str = "legacy"
    batch_size: int = 4
    num_workers: int = 20
    pin_memory: bool = True
    compile_tokenizer: bool = True
    max_save_workers: int = 16
    use_cuda: bool = torch.cuda.is_available()
    tokenizer_dtype: torch.dtype = torch.float16
    chunk_frames: int | None = 20 * 24_000  # None = variable-length (Phase 2+)
    num_channels: int = 1


class AudioChunkDataset(Dataset):
    """Audio file dataset. With chunk_frames=None we return variable-length tensors;
    DataLoader callers must use a custom collate or batch_size=1.
    """

    def __init__(
        self,
        input_path: str,
        target_sample_rate: int,
        num_channels: int,
        dtype: torch.dtype,
        chunk_frames: int | None = None,
    ) -> None:
        self.input_path = input_path
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames
        self.num_channels = num_channels
        self.dtype = dtype
        self.file_list: list[str] = []
        self._find_audio_files(input_path)

    def _find_audio_files(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(
                    (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".webm")
                ):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, int]:
        file_path = self.file_list[idx]
        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception:
            # Corrupted file — return tiny silence; caller filters by original_samples=0
            waveform = torch.zeros(self.num_channels, 1, dtype=self.dtype)
            sample_rate = self.target_sample_rate

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Adjust channel count
        if waveform.shape[0] < self.num_channels:
            waveform = waveform.repeat(self.num_channels, 1)
        elif waveform.shape[0] > self.num_channels:
            waveform = waveform[: self.num_channels]

        original_samples = int(waveform.shape[1])

        if self.chunk_frames is not None:
            # Legacy path: pad/truncate to fixed length (Phase 1a still uses this on
            # the existing 20s YT chunks). Phase 1.5 derives a valid_token_mask from
            # `original_samples` so the silent tail is masked at train time.
            if waveform.shape[1] < self.chunk_frames:
                waveform = torch.nn.functional.pad(
                    waveform, (0, self.chunk_frames - waveform.shape[1])
                )
            elif waveform.shape[1] > self.chunk_frames:
                waveform = waveform[:, : self.chunk_frames]

        waveform = waveform.to(self.dtype)
        return waveform, os.path.basename(self.file_list[idx]), original_samples


def _create_dataloader(config: PreprocessingConfig) -> DataLoader:
    dataset = AudioChunkDataset(
        config.input_path,
        target_sample_rate=24_000,
        chunk_frames=config.chunk_frames,
        num_channels=config.num_channels,
        dtype=config.tokenizer_dtype,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _save_tokens_v2(
    *,
    tokens: torch.Tensor,
    metadata: dict,
    filename: str,
    output_path: str,
    valid_token_mask: list[bool] | None = None,
    run_id: str | None = None,
):
    """Write `<filename>.npy` + `<filename>.meta.json` atomically under output_path.

    Uses two-char hash sharding (matches the existing layout under data/tokens/).
    """
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)
    base = os.path.splitext(filename)[0]
    npy_target = os.path.join(full_output_path, f"{base}.npy")
    atomic_save_npy(tokens.detach().cpu().numpy(), npy_target)
    write_metadata_sidecar(
        npy_target,
        metadata,
        valid_token_mask=valid_token_mask,
        run_id=run_id,
    )


def preprocess_audio_chunks(config: PreprocessingConfig):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path, exist_ok=True)

    dataloader = _create_dataloader(config)

    device = torch.device("cuda" if config.use_cuda else "cpu")
    tokenizer = VoxtralTokenizer(config.voxtral_tokenizer_config).to(
        device=device, dtype=config.tokenizer_dtype
    )
    if config.compile_tokenizer:
        tokenizer = typing.cast(
            VoxtralTokenizer, torch.compile(tokenizer, mode="reduce-overhead")
        )

    save_executor = ThreadPoolExecutor(max_workers=config.max_save_workers)
    run_id = new_run_id()

    skipped = 0
    for batch in tqdm(dataloader, desc="Processing audio chunks"):
        waveforms, filenames, original_samples = batch
        try:
            encoded, meta = tokenizer.encode(
                waveforms,
                24_000,
                language=config.language,
                source_metadata={"source": config.source},
            )
        except Exception as e:
            skipped += len(filenames)
            print(f"Skipped {len(filenames)} files due to error: {e}")
            continue

        for i, (z, filename) in enumerate(zip(encoded, filenames)):
            n_real = int(original_samples[i])
            chunk_frames = config.chunk_frames or n_real
            mask = derive_valid_token_mask_from_audio_length(
                audio_samples=n_real,
                chunk_samples=chunk_frames,
                token_count=int(z.numel()),
                stride=tokenizer.stream_stride if hasattr(tokenizer, "stream_stride") else 21,
            )
            # batch>1: meta is from batch[0]'s transcript only — fine for legacy path
            # (callers that need per-item transcripts use retokenize_v2.py with batch=1).
            per_item_meta = dict(meta)
            per_item_meta["language"] = config.language
            save_executor.submit(
                _save_tokens_v2,
                tokens=z,
                metadata=per_item_meta,
                filename=filename,
                output_path=config.output_path,
                valid_token_mask=mask,
                run_id=run_id,
            )
    if skipped:
        print(f"Total skipped: {skipped} files")

    save_executor.shutdown(wait=True)


# Back-compat shim — old _save_tokens(encoded, filename, path) signature kept
# for any external script that still imports it. Internally it builds an empty
# metadata dict (sufficient for non-v2 paths).
def _save_tokens(encoded_tokens: np.ndarray, filename: str, output_path: str):
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)
    output_file = os.path.join(full_output_path, f"{os.path.splitext(filename)[0]}.npy")
    arr = encoded_tokens.cpu().numpy() if isinstance(encoded_tokens, torch.Tensor) else encoded_tokens
    atomic_save_npy(np.asarray(arr), output_file)


if __name__ == "__main__":
    config = PreprocessingConfig()
    preprocess_audio_chunks(config)
