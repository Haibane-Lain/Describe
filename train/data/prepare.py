"""Build unified manifests from raw datasets.

Produces one JSONL record per image, so downstream stages never need to know
which dataset a row came from.

Record schema::

    {
      "image_id": "vizwiz_train_00023410",
      "file_name": "VizWiz_train_00023410.jpg",
      "dataset": "vizwiz" | "flickr30k",
      "split": "train" | "val" | "test",
      "captions": ["...", ...],
      "text_detected": true | false | null,
      "quality": {
        "unrecognizable_votes": 4,       # out of 5
        "unrecognizable": true,          # majority rule
        "flaws": {"BLR": 5, "DRK": 0, ...}
      } | null
    }

    py -3.12 -m train.data.prepare --dataset all
    py -3.12 -m train.data.prepare --dataset vizwiz --check-leakage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FLAW_CODES = ("BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT")
UNRECOGNIZABLE_MAJORITY = 3  # out of 5 votes


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_first(root: Path, patterns: list[str]) -> Path | None:
    """Return the first path matching any glob pattern."""
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def index_images(image_root: Path) -> dict[str, Path]:
    """Map bare filename -> path for every image under a root."""
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[path.name] = path
    return index


# ---------------------------------------------------------------- vizwiz
def build_vizwiz(
    data_root: Path, out_dir: Path, check_leakage: bool
) -> list[dict]:
    vizwiz = data_root / "vizwiz"
    image_index = index_images(vizwiz / "images")
    print(f"  indexed {len(image_index):,} VizWiz images")

    captions_by_image: dict[str, dict] = {}
    caption_root = vizwiz / "annotations" / "caption"
    if caption_root.exists():
        for path in sorted(caption_root.rglob("*.json")):
            payload = load_json(path)
            split = _split_from_name(path.name)
            for image in payload.get("images", []):
                captions_by_image[image["file_name"]] = {
                    "split": split,
                    "text_detected": image.get("text_detected"),
                    "captions": [],
                }
            for ann in payload.get("annotations", []):
                if ann.get("is_rejected") or ann.get("is_precanned"):
                    continue
                # Match annotation to image by id via the images list.
                for image in payload.get("images", []):
                    if image.get("id") == ann.get("image_id"):
                        entry = captions_by_image.get(image["file_name"])
                        if entry is not None:
                            entry["captions"].append(ann.get("caption", ""))
                            # The per-image flag is missing for many images, but
                            # each caption carries its own flag. Any caption
                            # reporting text means the image contains text.
                            flag = ann.get("text_detected")
                            if flag:
                                entry["text_detected"] = True
                            elif entry.get("text_detected") is None:
                                entry["text_detected"] = False
                        break
        detected = sum(
            1 for e in captions_by_image.values() if e.get("text_detected") is True
        )
        print(f"  caption annotations for {len(captions_by_image):,} images")
        print(f"  text_detected: {detected:,} with text, "
              f"{len(captions_by_image) - detected:,} without")
    else:
        print(f"  WARNING: no caption annotations at {caption_root}")

    quality_by_image: dict[str, dict] = {}
    quality_root = vizwiz / "annotations" / "quality"
    if quality_root.exists():
        for path in sorted(quality_root.rglob("*.json")):
            payload = load_json(path)
            for entry in payload if isinstance(payload, list) else payload.get("data", []):
                filename = entry.get("image")
                if not filename:
                    continue
                votes = int(entry.get("unrecognizable", 0) or 0)
                quality_by_image[filename] = {
                    "unrecognizable_votes": votes,
                    "unrecognizable": votes >= UNRECOGNIZABLE_MAJORITY,
                    "flaws": {code: int(entry.get("flaws", {}).get(code, 0) or 0)
                              for code in FLAW_CODES},
                }
        print(f"  quality annotations for {len(quality_by_image):,} images")
    else:
        print(f"  WARNING: no quality annotations at {quality_root}")

    records: list[dict] = []
    missing_images = 0
    for filename, caption_entry in captions_by_image.items():
        if filename not in image_index:
            missing_images += 1
            continue
        records.append(
            {
                "image_id": Path(filename).stem,
                "file_name": filename,
                "dataset": "vizwiz",
                "split": caption_entry["split"],
                "captions": [c for c in caption_entry["captions"] if c],
                "text_detected": caption_entry["text_detected"],
                "quality": quality_by_image.get(filename),
            }
        )

    if missing_images:
        print(f"  WARNING: {missing_images:,} annotated images not found on disk")

    if check_leakage:
        _report_leakage(records)

    _write_manifests(records, out_dir, "vizwiz")
    return records


def _split_from_name(name: str) -> str:
    lowered = name.lower()
    for split in ("train", "val", "test"):
        if split in lowered:
            return "val" if split == "val" else split
    return "train"


def _report_leakage(records: list[dict]) -> None:
    """Verify no image appears in more than one split.

    VizWiz-Captions and VizWiz-QualityIssues share an image pool and partition.
    Mixing splits between the two tasks is the most likely silent evaluation bug
    in this project, so it is asserted rather than assumed.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for record in records:
        key = record["file_name"]
        previous = seen.get(key)
        if previous is not None and previous != record["split"]:
            collisions.append(f"{key}: {previous} vs {record['split']}")
        seen[key] = record["split"]

    if collisions:
        print(f"  LEAKAGE: {len(collisions)} images appear in multiple splits")
        for line in collisions[:5]:
            print(f"    {line}")
    else:
        counts = Counter(r["split"] for r in records)
        summary = ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))
        print(f"  split integrity OK ({summary})")


# -------------------------------------------------------------- flickr30k
def build_flickr30k(data_root: Path, out_dir: Path) -> list[dict]:
    root = data_root / "flickr30k"
    captions_path = root / "captions.jsonl"
    if not captions_path.exists():
        print(f"  WARNING: {captions_path} not found; run download first")
        return []

    records: list[dict] = []
    with captions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "image_id": row.get("image_id"),
                    "file_name": row["file_name"],
                    "dataset": "flickr30k",
                    "split": "train",
                    "captions": row.get("captions", []),
                    "text_detected": None,
                    "quality": None,
                }
            )

    empty = sum(1 for r in records if not r["captions"])
    if empty:
        print(f"  WARNING: {empty:,} records have no captions")
    print(f"  flickr30k records: {len(records):,}")

    _write_manifests(records, out_dir, "flickr30k")
    return records


# ------------------------------------------------------------------- output
def _write_manifests(records: list[dict], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / f"{name}.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"  wrote {combined} ({len(records):,} records)")

    by_split: dict[str, list[dict]] = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)
    for split, rows in sorted(by_split.items()):
        path = out_dir / f"{name}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(record) + "\n")
        print(f"    {split:5s}: {len(rows):,} -> {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset manifests")
    parser.add_argument("--dataset", choices=["flickr30k", "vizwiz", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="assert no image spans multiple splits",
    )
    args = parser.parse_args()

    print(f"Data root: {args.data_root.resolve()}")
    print(f"Manifests: {args.out_dir.resolve()}")

    if args.dataset in ("vizwiz", "all"):
        print("\nVizWiz")
        build_vizwiz(args.data_root, args.out_dir, args.check_leakage)
    if args.dataset in ("flickr30k", "all"):
        print("\nFlickr30k")
        build_flickr30k(args.data_root, args.out_dir)

    print("\nNext: py -3.12 -m train.data.vocab is built automatically during training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
