"""Verify that an expanded caption index shares feature rows correctly.

Phase 2 relies on one property: many index rows point at the same `feature_row`,
so two rows for the same image must yield **identical image features** and
**different captions**. If the dataset re-sampled a caption, or if the shared row
were resolved wrongly, the expansion would silently do nothing (or worse, train
on mismatched image-text pairs).

Run directly:  py -3.12 train/finetune/_verify_expanded.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer  # noqa: E402

from train.finetune.cache import CachePaths, load_cache  # noqa: E402
from train.finetune.dataset import CachedBlipDataset, build_split, captions_of  # noqa: E402

EXPANDED = Path("data/features/vizwiz_train_large_expanded.jsonl")
TENSOR = Path("data/features/vizwiz_train_large_blip.npy")
TOKENIZER = "Salesforce/blip-image-captioning-large"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    features, index = load_cache(CachePaths(index=EXPANDED, tensor=TENSOR))
    rows = build_split(index)

    print(f"  expanded index rows   : {len(index):,}")
    print(f"  trainable rows        : {len(rows):,}")
    print(f"  tensor rows           : {features.shape[0]:,}")
    print(f"  tensor shape          : {tuple(features.shape)}")
    print()

    distinct = {int(r["feature_row"]) for r in rows}
    check("index is larger than the tensor (shared rows)",
          len(rows) > features.shape[0],
          f"{len(rows):,} rows over {features.shape[0]:,} feature rows")
    # 565 training images have no captions, so their tensor rows are legitimately
    # never referenced. Requiring every row to be referenced would fail.
    check("every referenced row is in range",
          max(distinct) < features.shape[0],
          f"max feature_row {max(distinct):,} < {features.shape[0]:,}")
    check("no image is referenced that should have no captions",
          len(distinct) == 22866,
          f"{len(distinct):,} distinct rows (expected 22,866 captioned images)")

    captions = [captions_of(r) for r in rows]
    check("every row carries exactly one caption",
          all(len(c) == 1 for c in captions),
          f"{sum(1 for c in captions if len(c) == 1):,}/{len(captions):,}")
    check("no row has zero captions",
          all(len(c) > 0 for c in captions))

    # The core property: shared feature row -> identical embeds, different caption.
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dataset = CachedBlipDataset(rows[:4000], features, tokenizer, max_length=32, seed=0)

    by_row: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(rows[:4000]):
        by_row[int(row["feature_row"])].append(position)
    pair = next((v for v in by_row.values() if len(v) > 1), None)
    if pair is None:
        check("found two rows sharing a feature row", False, "no shared row in the sample")
    else:
        a, b = dataset[pair[0]], dataset[pair[1]]
        check("shared feature row yields identical image features",
              bool((a["image_embeds"] == b["image_embeds"]).all()))
        # Captions differ in length, so compare token lists rather than tensors.
        tokens_a = a["input_ids"].tolist()
        tokens_b = b["input_ids"].tolist()
        check("the two rows have different captions",
              tokens_a != tokens_b,
              f"{captions_of(rows[pair[0]])[0][:32]!r} vs {captions_of(rows[pair[1]])[0][:32]!r}")

    # Distribution: how many captions per image did the expansion actually recover?
    per_image = defaultdict(int)
    for row in rows:
        per_image[row.get("_source_image")] += 1
    counts = sorted(per_image.values())
    check("expansion recovered multiple captions per image",
          counts[len(counts) // 2] >= 2,
          f"median {counts[len(counts) // 2]} captions/image, max {counts[-1]}")

    # Feature values must actually vary between images (not a constant block).
    sample = np.stack([np.asarray(features[int(r["feature_row"])], dtype=np.float32)
                       for r in rows[:64]])
    check("features vary across images",
          float(sample.std()) > 0.01,
          f"std {sample.std():.4f}")

    print()
    if failures:
        print(f"  {len(failures)} CHECK(S) FAILED: {failures}")
        return 1
    print("  all checks passed - the expanded index is safe to train on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
