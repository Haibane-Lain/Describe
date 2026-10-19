"""Extract a feature-cache subset by image name.

The VizWiz-VQA informative slice is 5,999 images that are already in the VizWiz
training split, so their vision features are already cached - no re-encoding is
needed. What *is* needed is a cache containing only those rows, with
``feature_row`` remapped to the new tensor, so it can be merged with the clean
cache for a mixed training run.

    py -3.12 -m train.finetune.subset_cache \
        --source-index data/features/vizwiz_train_full_blip.jsonl \
        --select data/manifests/vqa_informative_train.jsonl \
        --out-index data/features/vizwiz_vqa_slice_blip.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a cache subset by image name.")
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-tensor", type=Path, default=None,
                        help="defaults to the index path with .npy")
    parser.add_argument("--select", type=Path, required=True,
                        help="JSONL whose file_name values define the subset")
    parser.add_argument("--out-index", type=Path, required=True)
    parser.add_argument("--out-tensor", type=Path, default=None,
                        help="defaults to the output index path with .npy")
    parser.add_argument("--carry-captions", action="store_true",
                        help="use the selecting file's captions instead of the source's")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    source_tensor = args.source_tensor or args.source_index.with_suffix(".npy")
    out_tensor = args.out_tensor or args.out_index.with_suffix(".npy")

    from train.finetune.cache import row_image_key

    source_rows = load_jsonl(args.source_index)
    select_rows = load_jsonl(args.select)
    by_name = {}
    for row in source_rows:
        key = row_image_key(row)
        if key:
            by_name.setdefault(key, row)

    wanted = [row_image_key(r) for r in select_rows]
    wanted = [w for w in wanted if w]

    missing = [w for w in wanted if w not in by_name]
    if missing:
        print(f"  ! {len(missing)} selected images are not in the source cache "
              f"(first: {missing[:3]})")

    array = np.load(source_tensor, mmap_mode="r")
    out_rows: list[dict] = []
    indices: list[int] = []
    seen: set[str] = set()
    for name in wanted:
        if name in seen or name not in by_name:
            continue
        seen.add(name)
        indices.append(int(by_name[name]["feature_row"]))
        out_rows.append(by_name[name])

    if not out_rows:
        raise SystemExit("nothing selected; check that the file_name values match the source cache")

    out_tensor.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        out_tensor, mode="w+", dtype=np.float16, shape=(len(out_rows), *array.shape[1:])
    )
    block = 2048
    for start in range(0, len(indices), block):
        chunk = np.asarray(array[indices[start : start + block]], dtype=np.float16)
        out[start : start + chunk.shape[0]] = chunk
    out.flush()
    del out

    select_by_name = {row_image_key(r): r for r in select_rows}
    with args.out_index.open("w", encoding="utf-8") as handle:
        for new_row, entry in enumerate(out_rows):
            record = dict(entry)
            record["feature_row"] = new_row
            record["feature_file"] = out_tensor.name
            record["source_cache"] = args.source_index.name
            if args.carry_captions:
                replacement = select_by_name.get(row_image_key(entry)) or {}
                if replacement.get("captions"):
                    record["captions"] = replacement["captions"]
                    record["caption_source"] = replacement.get("source")
                    record["synthetic_caption"] = bool(replacement.get("synthetic_caption"))
            handle.write(json.dumps(record) + "\n")

    print(f"  source : {args.source_index.name}  ({len(source_rows):,} rows)")
    print(f"  select : {args.select.name}  ({len(select_rows):,} rows)")
    print(f"  subset : {len(out_rows):,} unique rows -> shape ({len(out_rows)}, "
          f"{', '.join(map(str, array.shape[1:]))})")
    print(f"  written: {out_tensor}")
    print(f"  written: {args.out_index}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "source_index": str(args.source_index),
                    "select": str(args.select),
                    "rows": len(out_rows),
                    "missing_from_source": len(missing),
                    "carry_captions": args.carry_captions,
                    "out_index": str(args.out_index),
                    "out_tensor": str(out_tensor),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
