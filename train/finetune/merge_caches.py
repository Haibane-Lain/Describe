"""Concatenate feature caches, so one training run can mix clean and degraded data.

The degradation experiment needs a model trained on a **mix** of clean and
degraded images in the same epoch. Training on degraded images alone would trade
clean performance for degraded performance, which is not a result - the
acceptance criterion requires that the recognizable-image metrics hold while the
unrecognizable ones improve.

Rather than teach the training loop to sample from several caches, the caches are
concatenated into one and the existing dataset code handles it unchanged. Row
identity is preserved: the merged index carries every source field plus
`degraded_with`, so a training run can be audited afterwards.

Merging duplicates images by design - a clean row and its degraded counterpart
are two rows over the same photograph - which is exactly the 50/50 mix wanted.

    py -3.12 -m train.finetune.merge_caches \
        --cache data/features/vizwiz_train_full_blip.jsonl \
        --cache data/features/vizwiz_train_degraded_blip.jsonl \
        --out-index data/features/vizwiz_train_mixed_blip.jsonl \
        --out-tensor data/features/vizwiz_train_mixed_blip.npy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_index(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate feature caches.")
    parser.add_argument("--cache", type=Path, nargs="+", required=True,
                        help="cache index files, in order; each must have a matching .npy")
    parser.add_argument("--out-index", type=Path, required=True)
    parser.add_argument("--out-tensor", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    arrays = []
    merged: list[dict] = []
    per_source: list[dict] = []

    for index_path in args.cache:
        tensor_path = index_path.with_suffix(".npy")
        if not tensor_path.exists():
            raise SystemExit(f"missing tensor for {index_path}: expected {tensor_path}")
        rows = load_index(index_path)
        array = np.load(tensor_path, mmap_mode="r")
        if array.shape[0] != len(rows):
            raise SystemExit(
                f"{index_path}: {len(rows)} index rows but {array.shape[0]} tensor rows"
            )

        base = len(merged)
        degraded = sum(1 for r in rows if r.get("degraded_with"))
        for row in rows:
            entry = dict(row)
            entry["feature_row"] = base + int(row["feature_row"])
            entry["feature_file"] = args.out_tensor.name
            entry["source_cache"] = index_path.name
            merged.append(entry)
        arrays.append(np.asarray(array))

        per_source.append(
            {
                "index": str(index_path),
                "rows": len(rows),
                "degraded_rows": degraded,
                "shape": list(array.shape),
            }
        )
        print(f"  {index_path.name:<44} {len(rows):>7} rows  shape={tuple(array.shape)}"
              + (f"  degraded={degraded}" if degraded else ""))

    shapes = {tuple(a.shape[1:]) for a in arrays}
    if len(shapes) != 1:
        raise SystemExit(
            f"caches have incompatible feature shapes {sorted(shapes)}; they must come from "
            "the same model and image size"
        )

    total_rows = sum(a.shape[0] for a in arrays)
    feature_shape = arrays[0].shape[1:]
    args.out_tensor.parent.mkdir(parents=True, exist_ok=True)

    # Stream into a memory-mapped output rather than concatenating in RAM.
    # VizWiz train is 23,431 x 577 x 768 float16 per cache, so the merged array
    # is ~39 GB - more than this machine's 31 GB of RAM, and np.concatenate
    # raises `Unable to allocate 38.7 GiB`. Writing row-block by row-block keeps
    # peak memory to one block regardless of corpus size.
    out = np.lib.format.open_memmap(
        args.out_tensor, mode="w+", dtype=np.float16, shape=(total_rows, *feature_shape)
    )
    block = 2048
    written = 0
    for array in arrays:
        for start in range(0, array.shape[0], block):
            chunk = np.asarray(array[start : start + block], dtype=np.float16)
            out[written : written + chunk.shape[0]] = chunk
            written += chunk.shape[0]
    out.flush()
    del out

    with args.out_index.open("w", encoding="utf-8") as handle:
        for entry in merged:
            handle.write(json.dumps(entry) + "\n")

    total_degraded = sum(s["degraded_rows"] for s in per_source)
    print(f"\n  merged: {len(merged):,} rows, shape=({total_rows}, {', '.join(map(str, feature_shape))})")
    print(f"  degraded rows: {total_degraded:,} ({100 * total_degraded / max(len(merged), 1):.1f}%)")
    print(f"  written: {args.out_tensor}")
    print(f"  written: {args.out_index}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "output_index": str(args.out_index),
                    "output_tensor": str(args.out_tensor),
                    "rows": len(merged),
                    "degraded_rows": total_degraded,
                    "degraded_fraction": total_degraded / max(len(merged), 1),
                    "sources": per_source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
