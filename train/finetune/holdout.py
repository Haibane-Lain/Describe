"""The selection/reporting split, defined in one place.

Per-epoch checkpoint selection uses a cheap 1,500-image subset. Reporting must
not use those same images, or the reported metrics are optimistically biased by
selection - the model was chosen partly because it happened to score well on
them.

That flaw was present in every table produced before this module existed: the
1,500-image selection subset was fully contained in the 7,542 images reported on.
The fix is not to stop selecting (selection is necessary) but to make the two
sets disjoint, and to derive both from one definition so they cannot drift apart.

Definitions:

* **selection** - 1,500 scorable validation images, drawn at random with seed 0.
  Used for per-epoch checkpoint choice and for nothing else.
* **reported** - the remaining 6,042 scorable images. Every number that appears
  in a table, a README, or a report comes from this set.

Both are derived from the validation cache index, so they are reproducible on any
machine that has the same cache.
"""

from __future__ import annotations

from pathlib import Path

SELECTION_SIZE = 1500
SELECTION_SEED = 0

# Cached once per process: building the split walks the whole cache index.
_SELECTION_NAMES: set[str] | None = None


def selection_names(
    cache_index: Path = Path("data/features/vizwiz_val_full_blip.jsonl"),
    cache_tensor: Path = Path("data/features/vizwiz_val_full_blip.npy"),
) -> set[str]:
    """File names of the 1,500 images used for checkpoint selection."""
    global _SELECTION_NAMES
    if _SELECTION_NAMES is not None:
        return _SELECTION_NAMES

    from train.finetune.cache import CachePaths, load_cache, row_image_key
    from train.finetune.dataset import build_split

    _, index = load_cache(CachePaths(index=cache_index, tensor=cache_tensor))
    rows = build_split(index, keep=SELECTION_SIZE, sample=True, seed=SELECTION_SEED)
    _SELECTION_NAMES = {row_image_key(row) for row in rows}
    return _SELECTION_NAMES


def partition(
    rows: list[dict],
    *,
    cache_index: Path = Path("data/features/vizwiz_val_full_blip.jsonl"),
    cache_tensor: Path = Path("data/features/vizwiz_val_full_blip.npy"),
) -> tuple[list[dict], list[dict]]:
    """Split rows into (selection, reported) by file name.

    Returns ``(selection_rows, reported_rows)``. The reported rows are the ones
    that must be used for any published figure.
    """
    from train.finetune.cache import row_image_key

    names = selection_names(cache_index, cache_tensor)
    selection = [row for row in rows if row_image_key(row) in names]
    reported = [row for row in rows if row_image_key(row) not in names]
    return selection, reported


def filter_predictions(predictions: list[dict], *, invert: bool = False) -> list[dict]:
    """Keep only predictions whose image is in (or is not in) the selection set.

    ``invert=False`` (default) returns the **reported** rows: images that were
    never used to choose a checkpoint.
    """
    names = selection_names()
    if invert:
        return [p for p in predictions if p.get("file_name") in names]
    return [p for p in predictions if p.get("file_name") not in names]


def describe(
    cache_index: Path = Path("data/features/vizwiz_val_full_blip.jsonl"),
    cache_tensor: Path = Path("data/features/vizwiz_val_full_blip.npy"),
) -> str:
    """One-line summary, for printing at the top of an evaluation run."""
    from train.finetune.cache import CachePaths, load_cache
    from train.finetune.dataset import build_split

    _, index = load_cache(CachePaths(index=cache_index, tensor=cache_tensor))
    total = len(build_split(index))
    selection = len(selection_names(cache_index, cache_tensor))
    return (
        f"validation split: {total} scorable images -> "
        f"{selection} for checkpoint selection, {total - selection} reported"
    )


__all__ = [
    "SELECTION_SEED",
    "SELECTION_SIZE",
    "describe",
    "filter_predictions",
    "partition",
    "selection_names",
]
