"""Turn COCO 2014 into the project's manifest format.

Produces the same JSONL shape the rest of the pipeline already consumes
(``image_id``, ``file_name``, ``captions``), so nothing downstream changes.

Two splits are supported:

* **Karpathy** (preferred, used in most captioning papers) — needs
  ``dataset_coco.json``. If it is not present this falls back to
* **standard COCO** — captions from ``captions_train2014.json`` and
  ``captions_val2014.json``, with the official train/val image split.

Karpathy's file is not part of the official COCO archives, so it is looked for
at ``data/raw/coco/annotations/dataset_coco.json`` and, if missing, an attempt
is made to fetch it from a known mirror.

    py -3.12 -m train.data.prepare_coco
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.download import COCO_KARPATHY_TRAIN, COCO_KARPATHY_VAL  # noqa: E402

# Karpathy's split file is mirrored in several places; try them in order.
KARPATHY_URLS = (
    "https://huggingface.co/datasets/nlphuji/coco_karpathy/resolve/main/dataset_coco.json",
    "https://raw.githubusercontent.com/karpathy/neuraltalk2/master/data/dataset_coco.json",
)


def fetch_karpathy(dest: Path) -> Path | None:
    """Download ``dataset_coco.json`` if it is not already present."""
    if dest.exists():
        return dest
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in KARPATHY_URLS:
        try:
            print(f"  fetching Karpathy split from {url}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            data = json.loads(payload)
            if "images" not in data:
                print("  unexpected schema; skipping this mirror")
                continue
            dest.write_bytes(payload)
            print(f"  saved {dest} ({len(payload) / 1e6:.1f} MB)")
            return dest
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {str(exc)[:80]}")
    return None


def index_images(images_root: Path) -> dict[str, Path]:
    """Map file name -> path across train2014/ and val2014/."""
    index: dict[str, Path] = {}
    for sub in ("train2014", "val2014"):
        directory = images_root / sub
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if path.suffix.lower() == ".jpg":
                index[path.name] = path
    return index


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"  wrote {out} ({len(rows):,} images)")


def from_karpathy(karpathy: Path, index: dict[str, Path]) -> dict[str, list[dict]]:
    """Split Karpathy annotations into train/val/test manifests."""
    data = json.loads(karpathy.read_text(encoding="utf-8"))
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing = 0
    for item in data["images"]:
        split = item.get("split")
        if split not in splits:
            continue
        file_name = item.get("filename")
        if file_name not in index:
            missing += 1
            continue
        captions = [
            sentence["raw"].strip()
            for sentence in item.get("sentences", [])
            if sentence.get("raw", "").strip()
        ]
        if not captions:
            continue
        splits[split].append(
            {
                "image_id": str(item.get("cocoid", file_name)),
                "file_name": file_name,
                "captions": captions,
            }
        )
    if missing:
        print(f"  WARNING: {missing:,} Karpathy images not found on disk")
    return splits


def from_standard(
    annotations: Path, index: dict[str, Path]
) -> dict[str, list[dict]]:
    """Fall back to the official COCO train/val caption files."""
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    for split, name in (("train", "captions_train2014.json"),
                        ("val", "captions_val2014.json")):
        path = annotations / name
        if not path.exists():
            raise SystemExit(
                f"missing {path}; run download_coco() first"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        by_image: dict[int, list[str]] = {}
        for annotation in data["annotations"]:
            by_image.setdefault(annotation["image_id"], []).append(
                annotation["caption"].strip()
            )
        missing = 0
        for image in data["images"]:
            captions = [c for c in by_image.get(image["id"], []) if c]
            if not captions or image["file_name"] not in index:
                missing += 1
                continue
            splits[split].append(
                {
                    "image_id": str(image["id"]),
                    "file_name": image["file_name"],
                    "captions": captions,
                }
            )
        if missing:
            print(f"  WARNING: {split}: {missing:,} images skipped (no captions/not on disk)")
    return splits


def resolve_annotations(root: Path) -> Path:
    """Find where the annotation JSONs actually landed.

    ``annotations_trainval2014.zip`` contains a top-level ``annotations/``
    directory, so extracting it into ``annotations/`` produces the nested path
    ``annotations/annotations/``. Accept either layout.
    """
    if (root / "captions_train2014.json").exists():
        return root
    nested = root / "annotations"
    if (nested / "captions_train2014.json").exists():
        return nested
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare COCO manifests")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/coco"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--no-fetch", action="store_true",
                        help="do not try to download the Karpathy split")
    args = parser.parse_args()

    images_root = args.data_root / "images"
    annotations = resolve_annotations(args.data_root / "annotations")
    if not images_root.exists():
        raise SystemExit(f"no images at {images_root}; run download_coco() first")

    print("Indexing COCO images")
    index = index_images(images_root)
    print(f"  found {len(index):,} images")
    if not index:
        raise SystemExit("no .jpg files found; is the extraction complete?")

    karpathy = annotations / "dataset_coco.json"
    if not karpathy.exists() and not args.no_fetch:
        fetched = fetch_karpathy(karpathy)
        if fetched is None:
            print("  Karpathy split unavailable; using the standard COCO split")

    if karpathy.exists():
        print("\nUsing the Karpathy split")
        splits = from_karpathy(karpathy, index)
        expected = {"train": COCO_KARPATHY_TRAIN, "val": COCO_KARPATHY_VAL}
        for name, target in expected.items():
            got = len(splits.get(name, []))
            flag = "OK" if abs(got - target) < target * 0.05 else "UNEXPECTED"
            print(f"  {name}: {got:,} (expected ~{target:,}) [{flag}]")
    else:
        print("\nUsing the standard COCO split")
        splits = from_standard(annotations, index)

    for name, rows in splits.items():
        if rows:
            write_manifest(rows, args.out_dir / f"coco_{name}.jsonl")

    train_rows = splits.get("train", [])
    if train_rows:
        captions = sum(len(r["captions"]) for r in train_rows)
        print(f"\n  train captions: {captions:,} across {len(train_rows):,} images "
              f"({captions / len(train_rows):.1f} per image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
