"""Dual-stream tokenizer for OmniVoxtral.

Encodes two audio channels (user + model) into a single interleaved token sequence
for the dual-transformer architecture. During training, both channels come from
diarized conversational data. During inference, the user stream is live-encoded
and the model stream is generated.

Token layout per 200ms window (dual_stride=42):
    [user_text(1), user_audio(20), model_text(1), model_audio(20)]

Where each audio block contains interleaved codebook tokens:
    [q1_f1, q2_f1, ..., q8_f1, q1_f2, ..., q8_f2, q1_f3, ..., q4_f3]

This layout enables:
- Full duplex: model sees both streams simultaneously
- Natural turn-taking: learned from data (Fisher/IndicVoices), not from VAD
- Inner Monologue: text tokens for both user (transcription) and model (generation)
- Interruption handling: overlapping non-silence in both streams

Reference: Moshi paper Section 4.3, ARCHITECTURE.md Decision 3.
"""

import torch
import torchaudio as ta

import dotenv
import huggingface_hub as hf_hub

from .mimi.models import loaders
from .model import VoxtralTokenizerConfig, interleave, uninterleave
from .word_level_whisper import TimedWhisperTokenizer

dotenv.load_dotenv()


class DualStreamTokenizer(torch.nn.Module):
    """Dual-stream tokenizer for OmniVoxtral.

    Encodes two audio channels into a single interleaved sequence.
    Both channels use the same Mimi codec and Whisper text tokenizer.
    """

    def __init__(self, config: VoxtralTokenizerConfig):
        super().__init__()
        self.config = config

        # Mimi codec (shared for both streams)
        mimi_weight = hf_hub.hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device="cpu")
        self.mimi.set_num_codebooks(config.mimi_num_quantizers)

        self.text_to_audio_token_factor = int(
            self.mimi.frame_rate * config.mimi_num_quantizers / config.text_hz
        )

        # Whisper (shared for both streams)
        self.whisper = TimedWhisperTokenizer(
            config.whisper_path,
            hertz=config.text_hz,
            language=config.language,
            sp_tokenizer_path=config.sp_tokenizer_path,
        )

        # Stride calculations
        self.stream_stride = 1 + self.text_to_audio_token_factor  # 21 per stream
        self.dual_stride = 2 * self.stream_stride  # 42 total per window

    @property
    def device(self) -> torch.device:
        return next(self.mimi.parameters()).device

    def _encode_audio_tokens(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode raw audio to offset, interleaved codebook tokens.

        Args:
            audio: (batch, 1, samples) at 24kHz on self.device

        Returns:
            interleaved: (batch, num_audio_tokens) — codebook-interleaved, offset
        """
        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = torch.arange(0, self.config.mimi_num_quantizers) * mimi_vocab_size
        mimi_tokens = self.mimi.encode(audio)
        audio_tokens = (
            mimi_tokens
            + token_offset[None, :, None].to(audio.device)
            + self.config.text_vocab_size
        )
        return interleave(
            *audio_tokens.unbind(1),
            factors=[1] * self.config.mimi_num_quantizers,
        )

    def _encode_text_tokens(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode raw audio to text tokens via Whisper.

        Args:
            audio: (batch, 1, samples) at sample_rate

        Returns:
            text_tokens: (batch, num_text_frames)
        """
        x_16k = ta.functional.resample(audio, sample_rate, 16_000)[:, 0]
        return self.whisper(x_16k, sample_rate=16_000)

    @torch.no_grad()
    def encode(
        self,
        user_audio: torch.Tensor,
        model_audio: torch.Tensor,
        sample_rate: int,
    ) -> torch.Tensor:
        """Encode two audio streams into a dual-stream interleaved token sequence.

        Args:
            user_audio: (batch, 1, samples) — user's speech at 24kHz
            model_audio: (batch, 1, samples) — model's speech at 24kHz
            sample_rate: must be 24000

        Returns:
            tokens: (batch, seq_len) — dual-stream interleaved sequence
                Layout per window: [user_text(1), user_audio(20), model_text(1), model_audio(20)]
        """
        assert user_audio.dim() == 3 and user_audio.size(1) == 1
        assert model_audio.dim() == 3 and model_audio.size(1) == 1
        assert sample_rate == 24_000

        device = self.device

        # Encode both streams
        user_text = self._encode_text_tokens(user_audio, sample_rate)
        user_audio_tok = self._encode_audio_tokens(user_audio.to(device))
        model_text = self._encode_text_tokens(model_audio, sample_rate)
        model_audio_tok = self._encode_audio_tokens(model_audio.to(device))

        # Apply 2-window delay (text leads audio by 2 windows)
        delay_audio = int(self.config.mimi_num_quantizers * self.mimi.frame_rate * 2)
        delay_text = self.config.text_hz * 2

        user_audio_tok = user_audio_tok[..., delay_audio:]
        user_text = user_text[..., :-delay_text]
        model_audio_tok = model_audio_tok[..., delay_audio:]
        model_text = model_text[..., :-delay_text]

        # Trim to complete windows (all four tensors must align)
        atf = self.text_to_audio_token_factor
        num_windows = min(
            user_text.size(-1),
            model_text.size(-1),
            user_audio_tok.size(-1) // atf,
            model_audio_tok.size(-1) // atf,
        )

        user_text = user_text[..., :num_windows]
        user_audio_tok = user_audio_tok[..., : num_windows * atf]
        model_text = model_text[..., :num_windows]
        model_audio_tok = model_audio_tok[..., : num_windows * atf]

        # Build per-stream sequences: [text(1), audio(atf)] per window
        user_stream = interleave(user_text, user_audio_tok, factors=[1, atf])
        model_stream = interleave(model_text, model_audio_tok, factors=[1, atf])

        # Interleave the two streams: [user_window(21), model_window(21)] per window
        tokens = interleave(
            user_stream, model_stream,
            factors=[self.stream_stride, self.stream_stride],
        )

        return tokens

    def _decode_audio_from_tokens(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """Decode interleaved, offset audio tokens back to waveform.

        Args:
            audio_tokens: (batch, num_audio_tokens) — interleaved codebook tokens with offset

        Returns:
            audio: (batch, 1, samples)
        """
        audio_tokens = audio_tokens - self.config.text_vocab_size
        channels = uninterleave(
            audio_tokens, factors=[1] * self.config.mimi_num_quantizers
        )
        stacked = torch.stack(channels, dim=1)  # (batch, num_q, seq)

        mimi_vocab_size = self.mimi.quantizer.cardinality
        token_offset = (
            torch.arange(0, self.config.mimi_num_quantizers, device=stacked.device)
            * mimi_vocab_size
        )
        stacked = stacked - token_offset[None, :, None]
        stacked = torch.clamp(stacked, 0, mimi_vocab_size - 1)

        return self.mimi.decode(stacked)

    @torch.no_grad()
    def decode_model_stream(self, z: torch.Tensor) -> torch.Tensor:
        """Decode only the model stream from a dual-stream token sequence.

        Args:
            z: (batch, seq_len) — dual-stream interleaved sequence

        Returns:
            audio: (batch, 1, samples) — decoded model audio
        """
        _, model_stream = uninterleave(
            z, factors=[self.stream_stride, self.stream_stride]
        )
        _, model_audio = uninterleave(
            model_stream, factors=[1, self.text_to_audio_token_factor]
        )
        return self._decode_audio_from_tokens(model_audio)

    @torch.no_grad()
    def decode_user_stream(self, z: torch.Tensor) -> torch.Tensor:
        """Decode only the user stream (for debugging/evaluation).

        Args:
            z: (batch, seq_len) — dual-stream interleaved sequence

        Returns:
            audio: (batch, 1, samples) — decoded user audio
        """
        user_stream, _ = uninterleave(
            z, factors=[self.stream_stride, self.stream_stride]
        )
        _, user_audio = uninterleave(
            user_stream, factors=[1, self.text_to_audio_token_factor]
        )
        return self._decode_audio_from_tokens(user_audio)

    @torch.no_grad()
    def extract_text_tokens(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract text tokens from both streams (for evaluation/debugging).

        Returns:
            dict with 'user_text' and 'model_text', each (batch, num_windows)
        """
        user_stream, model_stream = uninterleave(
            z, factors=[self.stream_stride, self.stream_stride]
        )
        user_text, _ = uninterleave(
            user_stream, factors=[1, self.text_to_audio_token_factor]
        )
        model_text, _ = uninterleave(
            model_stream, factors=[1, self.text_to_audio_token_factor]
        )
        return {"user_text": user_text, "model_text": model_text}
