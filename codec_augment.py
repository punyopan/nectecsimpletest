"""
Codec Augmentation for voice anti-spoofing training.
Randomly encodes/decodes audio through lossy codecs (MP3, Opus)
to teach the model invariance to compression artifacts.
"""
import os
import random
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf


class CodecAugmentor:
    """Apply random lossy codec compression to audio during training.

    Supported codecs: mp3, opus (via ffmpeg).
    Falls back to bit-quantization noise if ffmpeg is unavailable.
    """

    DEFAULT_MP3_BITRATES = [64, 96, 128, 192, 256, 320]
    DEFAULT_OPUS_BITRATES = [16, 32, 48, 64, 96, 128]

    def __init__(
        self,
        codecs: Optional[List[str]] = None,
        prob: float = 0.5,
        mp3_bitrates: Optional[List[int]] = None,
        opus_bitrates: Optional[List[int]] = None,
    ):
        self.codecs = codecs or ["mp3", "opus"]
        self.prob = prob
        self.mp3_bitrates = mp3_bitrates or self.DEFAULT_MP3_BITRATES
        self.opus_bitrates = opus_bitrates or self.DEFAULT_OPUS_BITRATES
        self._ffmpeg_available = self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> bool:
        try:
            subprocess.run(["ffmpeg", "-version"],
                           capture_output=True, check=True, timeout=3)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError,
                subprocess.TimeoutExpired):
            return False

    def augment(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        if random.random() > self.prob:
            return waveform
        if not self._ffmpeg_available:
            return self._fallback_quantize(waveform)
        codec = random.choice(self.codecs)
        if codec == "mp3":
            return self._apply_codec(waveform, sr, "mp3",
                                     random.choice(self.mp3_bitrates))
        elif codec == "opus":
            return self._apply_codec(waveform, sr, "opus",
                                     random.choice(self.opus_bitrates))
        return waveform

    def _apply_codec(self, waveform, sr, fmt, bitrate):
        original_len = len(waveform)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = os.path.join(tmpdir, "input.wav")
                sf.write(input_path, waveform, sr)
                ext = "ogg" if fmt == "opus" else fmt
                encoded_path = os.path.join(tmpdir, f"encoded.{ext}")
                decoded_path = os.path.join(tmpdir, "decoded.wav")

                subprocess.run(
                    ["ffmpeg", "-y", "-i", input_path, "-acodec", fmt,
                     "-b:a", f"{bitrate}k", "-ar", str(sr), encoded_path],
                    capture_output=True, check=True, timeout=10)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", encoded_path, "-acodec",
                     "pcm_s16le", "-ar", str(sr), "-ac", "1", decoded_path],
                    capture_output=True, check=True, timeout=10)

                decoded, _ = sf.read(decoded_path, dtype="float32")
                if len(decoded) >= original_len:
                    return decoded[:original_len]
                padded = np.zeros(original_len, dtype=np.float32)
                padded[:len(decoded)] = decoded
                return padded
        except Exception:
            return waveform

    @staticmethod
    def _fallback_quantize(waveform: np.ndarray) -> np.ndarray:
        """Simulate lossy compression via bit-depth quantization."""
        bits = random.choice([6, 8, 10, 12])
        levels = 2 ** bits
        max_val = np.max(np.abs(waveform)) + 1e-8
        normalized = waveform / max_val
        quantized = np.round(normalized * (levels / 2)) / (levels / 2)
        return (quantized * max_val).astype(np.float32)
