"""Cached-feature dataset for decoder training.

Each row is one (image feature, caption) pair. An image with five captions
contributes five rows, which is standard for captioning and gives the decoder
more signal per epoch than sampling one caption at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from train.data.vocab import Vocab
from train.models.captioner import collate_captions


class FeatureCaptionDataset(Dataset):
    """Pairs cached CNN features with encoded captions.

    Works with either pooled ``[N, C]`` features or spatial ``[N, C, H, W]``
    maps. Spatial arrays should be opened memory-mapped, so a multi-gigabyte
    map is read lazily per item instead of being held in RAM.
    """

    def __init__(
        self,
        features: np.ndarray,
        rows: list[dict],
        vocab: Vocab,
        captions_per_image: int | None = None,
        seed: int = 1337,
        max_caption_len: int | None = None,
    ) -> None:
        self.features = features
        self.vocab = vocab
        self.seed = seed
        self.epoch = 0
        # Logits cost batch x seq_len x vocab, so the longest captions drive
        # peak memory. Capping the length bounds that; Flickr30k's median
        # caption is far below its maximum.
        self.max_caption_len = max_caption_len

        pairs: list[tuple[int, str]] = []
        for row in rows:
            captions = row.get("captions") or []
            if not captions:
                continue
            selected = captions
            if captions_per_image is not None:
                selected = captions[:captions_per_image]
            for caption in selected:
                encoded = vocab.encode(caption)
                # A caption of only <start>/<end> teaches nothing.
                if len(encoded) > 2:
                    pairs.append((int(row["feature_row"]), caption))

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[int]]:
        feature_row, caption = self.pairs[index]
        feature = torch.from_numpy(
            np.asarray(self.features[feature_row], dtype=np.float32)
        )
        encoded = self.vocab.encode(caption)
        if self.max_caption_len is not None and len(encoded) > self.max_caption_len:
            # Truncate but keep the trailing <end> so the model still learns
            # where captions stop.
            encoded = encoded[: self.max_caption_len - 1] + [encoded[-1]]
        return feature, encoded

    @property
    def num_images(self) -> int:
        return len({row for row, _ in self.pairs})


def load_feature_cache(
    index_path: Path,
    feature_path: Path | None = None,
    mmap: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Load a feature array and its index written by ``features.extract``.

    ``mmap=True`` is required for spatial maps: a 31,783-image
    ``[N, 2048, 7, 7]`` float16 array is ~6.4 GB, which should not be pulled
    into RAM when only one batch is needed at a time.
    """
    if feature_path is None:
        feature_path = index_path.with_suffix(".npy")
    if not feature_path.exists():
        raise FileNotFoundError(
            f"feature array not found: {feature_path}\n"
            "Run: py -3.12 -m train.features.extract --manifest <manifest>"
        )

    features = np.load(feature_path, mmap_mode="r" if mmap else None)
    rows = [
        json.loads(line)
        for line in index_path.open(encoding="utf-8")
        if line.strip()
    ]

    if len(rows) != features.shape[0]:
        raise ValueError(
            f"cache mismatch: {len(rows)} index rows vs {features.shape[0]} feature rows"
        )
    return features, rows


def make_collate(pad_idx: int):
    """Return a collate function closing over the pad index."""

    def collate(batch: list[tuple[torch.Tensor, list[int]]]):
        features = torch.stack([item[0] for item in batch])
        captions = collate_captions([item[1] for item in batch], pad_idx)
        return features, captions

    return collate
