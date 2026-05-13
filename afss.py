"""
Artifact-Focused Self-Synthesis (AFSS).
Generates 'pseudo-spoofs' by running bonafide speech through a vocoder
to force the model to learn vocoder artifacts instead of shortcuts.

For smoke tests, uses simple DSP distortion as mock vocoder.
"""
import random
from typing import Tuple

import numpy as np


class AFSSVocoder:
    """Wrapper for BigVGAN/HiFi-GAN vocoders for AFSS.
    In mock mode, applies simple DSP distortion instead.
    """
    def __init__(self, vocoder_type: str = "bigvgan", is_mock: bool = True):
        self.vocoder_type = vocoder_type
        self.is_mock = is_mock
        if not is_mock:
            print(f"Warning: Real {vocoder_type} not implemented. Using mock.")
            self.is_mock = True

    def synthesize(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process audio through vocoder to add synthesis artifacts."""
        if self.is_mock:
            # Mock: harmonic distortion + slight phase shift
            t = np.linspace(0, len(audio) / sr, len(audio))
            distortion = np.sin(2 * np.pi * 50 * t) * 0.05
            pseudo_spoof = audio + distortion
            return np.clip(pseudo_spoof, -1.0, 1.0).astype(np.float32)
        return audio


class AFSSAugmentor:
    """On-the-fly AFSS augmentor for training."""
    def __init__(self, vocoder_type: str = "bigvgan", prob: float = 0.5,
                 is_mock: bool = True):
        self.vocoder = AFSSVocoder(vocoder_type, is_mock=is_mock)
        self.prob = prob

    def augment(self, audio: np.ndarray, label: int,
                sr: int = 16000) -> Tuple[np.ndarray, int]:
        """If audio is bonafide (label=1), may convert to pseudo-spoof.
        Returns (audio, label) — label changes to 0 if spoofed.
        """
        if label == 1 and random.random() < self.prob:
            return self.vocoder.synthesize(audio, sr), 0
        return audio, label
