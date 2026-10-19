"""Expand a feature-cache index to one row per caption, sharing the same features.

The captioner holds **100,575 captions across 22,866 training images** (mean 4.4
each), but each epoch draws only *one* random caption per image. Over an 8-epoch
run the model therefore sees ~8 of a 4.4-caption pool, and each individual
caption only once or twice - thin exposure for a rare word.

This tool writes an index with one row per caption. Crucially it **does not
duplicate the feature tensor**: every row keeps the same ``feature_row``, and
``CachedBlipDataset`` indexes the shared array by that field, so 100,575 rows
point into the same 22,866 feature slots. The expanded index is a few tens of MB
where duplicating the tensor would need ~117 GB.

    py -3.12 -m train.finetune.expand_captions \
        --index data/features/vizwiz_train_large_blip.jsonl \
        --out-index data/features/vizwiz_train_large_expanded.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="One index row per caption, features shared.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out-index", type=Path, required=True)
    parser.add_argument("--max-captions-per-image", type=int, default=None,
                        help="cap captions per image; default uses all")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    from train.finetune.dataset import captions_of

    with args.index.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    out_rows: list[dict] = []
    images = 0
    images_without = 0
    for row in rows:
        captions = captions_of(row)
        if not captions:
            images_without += 1
            continue
        images += 1
        if args.max_captions_per_image:
            captions = captions[: args.max_captions_per_image]
        for caption in captions:
            # `captions` holds exactly one entry, and `_caption_single` tells the
            # dataset to use it directly rather than re-sampling from the list.
            # Without that flag the dataset would still pick one at random and
            # the expansion would achieve nothing.
            entry = dict(row)
            entry["captions"] = [caption]
            entry["_caption_single"] = True
            entry["_source_image"] = row.get("file_name") or row.get("image")
            out_rows.append(entry)

    args.out_index.parent.mkdir(parents=True, exist_ok=True)
    with args.out_index.open("w", encoding="utf-8") as handle:
        for entry in out_rows:
            handle.write(json.dumps(entry) + "\n")

    total_captions = sum(len(captions_of(r)) for r in rows)
    print(f"  source index      : {args.index.name}")
    print(f"  source rows       : {len(rows):,}  ({total_captions:,} captions held)")
    print(f"  images with caps  : {images:,}  (without: {images_without:,})")
    print(f"  expanded rows     : {len(out_rows):,}  "
          f"({len(out_rows) / max(images, 1):.2f}x)")
    print(f"  feature tensor    : reused unchanged - no duplication")
    print(f"  written           : {args.out_index}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "source_index": str(args.index),
                    "out_index": str(args.out_index),
                    "source_rows": len(rows),
                    "source_captions": total_captions,
                    "expanded_rows": len(out_rows),
                    "expansion_factor": len(out_rows) / max(images, 1),
                    "images_without_captions": images_without,
                    "feature_tensor_duplicated": False,
                    "note": (
                        "rows share feature_row values; the dataset indexes the same "
                        "feature array, so only the index grows"
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
