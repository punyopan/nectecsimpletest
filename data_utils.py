"""
Data utilities for spoof detection training.
Supports both ASVspoof-format and ThaiSpoof-format protocols.
"""
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """Parse ASVspoof-format protocol files.
    Format: SPEAKER_ID UTTERANCE_ID - ATTACK_TYPE LABEL
    """
    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def gen_universal_spoof_list(dir_meta, is_train=False, is_eval=False,
                              fmt_config=None):
    """Generic protocol parser for ThaiSpoof and other datasets.
    fmt_config keys: separator, key_col, label_col, lang_col, lang_filter
    """
    if fmt_config is None:
        fmt_config = {}

    sep = fmt_config.get("separator", " ")
    key_col = fmt_config.get("key_col", 1)
    label_col = fmt_config.get("label_col", 4)
    lang_col = fmt_config.get("lang_col", None)
    lang_filter = fmt_config.get("lang_filter", None)

    d_meta = {}
    file_list = []

    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        cols = line.strip().split(sep)
        if len(cols) <= max(key_col, label_col):
            continue

        key = cols[key_col]

        if lang_col is not None and lang_filter is not None:
            if len(cols) > lang_col and cols[lang_col] != lang_filter:
                continue

        file_list.append(key)

        if not is_eval:
            label = cols[label_col]
            d_meta[key] = 1 if label == "bonafide" else 0

    if is_eval:
        return file_list
    else:
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class SpoofDataset_train(Dataset):
    """Training dataset with optional codec and AFSS augmentation.
    Supports both .wav and .flac formats.
    """
    def __init__(self, list_IDs, labels, base_dir, audio_ext=".flac",
                 audio_subdir="flac", codec_augment=False, codec_config=None,
                 afss_augment=False, afss_config=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.audio_ext = audio_ext
        self.audio_subdir = audio_subdir
        self.cut = 64600  # ~4 sec at 16kHz

        # Codec augmentation
        self.codec_augment = codec_augment
        if self.codec_augment:
            from codec_augment import CodecAugmentor
            cfg = codec_config or {}
            self.codec_augmentor = CodecAugmentor(
                codecs=cfg.get("codecs", ["mp3", "opus"]),
                prob=cfg.get("prob", 0.5),
            )

        # AFSS augmentation
        self.afss_augment = afss_augment
        if self.afss_augment:
            from afss import AFSSAugmentor
            cfg = afss_config or {}
            self.afss_augmentor = AFSSAugmentor(
                prob=cfg.get("prob", 0.5),
                is_mock=cfg.get("is_mock", True),
            )

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if self.audio_subdir:
            file_path = str(
                self.base_dir / f"{self.audio_subdir}/{key}{self.audio_ext}")
        else:
            file_path = str(self.base_dir / f"{key}{self.audio_ext}")

        X, fs = sf.read(file_path)

        y = self.labels[key]

        # AFSS: may convert bonafide to pseudo-spoof
        if self.afss_augment and hasattr(self, 'afss_augmentor'):
            X, y = self.afss_augmentor.augment(X, y, sr=fs)

        # Codec augmentation
        if self.codec_augment and hasattr(self, 'codec_augmentor'):
            X = self.codec_augmentor.augment(X, sr=fs)

        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, y


class SpoofDataset_devNeval(Dataset):
    """Dev/eval dataset. No augmentation."""
    def __init__(self, list_IDs, base_dir, audio_ext=".flac",
                 audio_subdir="flac"):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.audio_ext = audio_ext
        self.audio_subdir = audio_subdir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        if self.audio_subdir:
            file_path = str(
                self.base_dir / f"{self.audio_subdir}/{key}{self.audio_ext}")
        else:
            file_path = str(self.base_dir / f"{key}{self.audio_ext}")

        X, _ = sf.read(file_path)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
