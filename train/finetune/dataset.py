"""Dataset and collator over cached BLIP image features.

Reads the cache index written by :mod:`train.finetune.cache`. Two decisions are
worth stating because they are the difference between a useful fine-tune and a
wasted one:

**One caption per image per epoch, sampled.** VizWiz images carry up to five
crowd-written captions. Cycling them deterministically means a 16-epoch run sees
the same five sentences 3.2 times each. Sampling with a seeded RNG means the
model sees a different sentence for most images every epoch - free augmentation
on a 23k-image corpus, which is the corpus size that hurt the from-scratch
model most.

**Pad is masked with -100.** ``BlipTextLMHeadModel`` computes its own loss with
label smoothing and shifts ``labels`` internally, so padding must be ``-100``
(``CrossEntropyLoss``'s ignore index), not the tokenizer's pad id.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_index(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"cache index not found: {path}. Run train.finetune.cache first.")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def captions_of(row: dict) -> list[str]:
    """Captions for a manifest row, tolerating the field spellings in this repo."""
    for key in ("captions", "caption", "sentences", "text"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    raw = item.get("raw") or item.get("caption") or item.get("text")
                    if isinstance(raw, str):
                        out.append(raw)
            return [c for c in out if c and c.strip()]
    return []


class CachedBlipDataset(Dataset):
    """Caption rows backed by cached image features.

    Args:
        index: rows from a cache index (already filtered to the split wanted).
        features: ``mmap_mode="r"`` array, ``[N, patches, hidden]`` float16.
        tokenizer: BLIP's tokenizer.
        max_length: caption truncation. VizWiz captions average ~11 tokens and
            the 99th percentile is well under 32; truncating at 32 costs nothing
            and bounds the ``[B, T, vocab]`` logits tensor, which is what
            exhausted memory in the from-scratch run.
        seed: base seed; each epoch derives its own sampling stream from it, so a
            resumed run reproduces the same caption sequence.
    """

    def __init__(
        self,
        index: list[dict],
        features: np.ndarray,
        tokenizer,
        *,
        max_length: int = 32,
        seed: int = 0,
    ) -> None:
        self.index = index
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        row = self.index[item]
        captions = captions_of(row)
        if not captions:
            # Should be filtered out at construction; fail loudly rather than
            # train on an empty target.
            raise SystemExit(f"row {item} has no captions: {row.get('image')!r}")

        # Deterministic per (epoch, row): reproducible, and different every epoch.
        #
        # Rows carrying `_caption_single` were produced by
        # `train.finetune.expand_captions`, which writes one row per caption so
        # that every caption in the corpus is trained on. Re-sampling randomly
        # here would pick the same one caption anyway, but skipping the draw
        # makes the intent explicit and keeps the expanded index meaningful.
        if row.get("_caption_single"):
            caption = captions[0]
        else:
            generator = np.random.default_rng((self.seed, self.epoch, item))
            caption = captions[int(generator.integers(len(captions)))]

        encoded = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = list(encoded["input_ids"])
        if not input_ids:
            input_ids = [self.tokenizer.bos_token_id or 0]

        row_index = int(row["feature_row"])
        # np.asarray on a memmapped cache yields a read-only array, which
        # torch warns about. Copy: the features are small per sample and the
        # warning is otherwise emitted on every batch.
        embeds = torch.from_numpy(
            np.array(self.features[row_index], dtype=np.float16, copy=True)
        )
        return {"image_embeds": embeds, "input_ids": torch.tensor(input_ids, dtype=torch.long)}


class BlipCaptionCollator:
    """Pad ``input_ids`` and build ``labels`` with -100 at pad positions.

    ``image_embeds`` are already uniform in shape, so they just stack.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        length = max(item["input_ids"].numel() for item in batch)

        input_ids = torch.full((len(batch), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), length), dtype=torch.long)

        for slot, item in enumerate(batch):
            ids = item["input_ids"]
            n = ids.numel()
            input_ids[slot, :n] = ids
            labels[slot, :n] = ids
            attention_mask[slot, :n] = 1

        embeds = torch.stack([item["image_embeds"] for item in batch]).to(torch.float32)
        return {
            "image_embeds": embeds,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_split(
    index: list[dict],
    *,
    keep: int | None = None,
    require_captions: bool = True,
    sample: bool = False,
    seed: int = 0,
    text_detected: bool | None = None,
) -> list[dict]:
    """Filter an index to trainable rows.

    Captionless rows are dropped: VizWiz has 565 train and 208 val images whose
    captions were all rejected, and training on an empty target teaches the
    model to emit nothing.

    ``keep`` truncates to the first N rows, which is what a smoke test wants.
    ``sample`` instead draws N rows at random with a fixed seed - what a
    learning-rate probe wants, because manifest order is annotation order and
    VizWiz photographs were collected in sessions. Taking the leading rows can
    therefore pick a cluster of one contributor's kitchen rather than a
    representative slice of blind-user photography.

    ``text_detected`` restricts to rows whose photographs do (or do not) contain
    text, using the per-image flag already in every manifest. This is how a
    text-focused variant is built without any new data: 16,272 of the 22,866
    captioned training images carry text.
    """
    rows = list(index)
    if require_captions:
        rows = [row for row in rows if captions_of(row)]
    if text_detected is not None:
        rows = [row for row in rows if row.get("text_detected") is text_detected]

    if keep is None:
        return rows
    if not sample or keep >= len(rows):
        return rows[:keep]

    generator = random.Random(seed)
    picked = set(generator.sample(range(len(rows)), keep))
    # Preserve manifest order so downstream code sees a stable sequence.
    return [row for position, row in enumerate(rows) if position in picked]


def describe_split(name: str, rows: list[dict]) -> str:
    caption_counts: dict[int, int] = {}
    for row in rows:
        n = len(captions_of(row))
        caption_counts[n] = caption_counts.get(n, 0) + 1
    spread = ", ".join(f"{n}cap:{c}" for n, c in sorted(caption_counts.items()))
    return f"{name}: {len(rows)} trainable rows ({spread})"


__all__ = [
    "IGNORE_INDEX",
    "BlipCaptionCollator",
    "CachedBlipDataset",
    "build_split",
    "captions_of",
    "describe_split",
    "load_index",
]
