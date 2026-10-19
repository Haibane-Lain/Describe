"""Collect the measured facts about this project's datasets.

Prints a JSON blob on stdout; nothing is written to disk. The numbers feed the
dataset write-up, so every value here is measured rather than asserted.
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT: dict[str, object] = {}


def count_dir(path: Path, suffix: str = ".jpg") -> int:
    if not path.is_dir():
        return -1
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() == suffix)


def dir_bytes(path: Path) -> float:
    total = 0
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return round(total / 1e9, 2)


# ---------------------------------------------------------------- VizWiz-Captions
cap: dict[str, object] = {}
for split in ("train", "val", "test"):
    fp = BASE / "data/raw/vizwiz/annotations/caption/annotations" / f"{split}.json"
    if not fp.exists():
        continue
    d = json.loads(fp.read_text(encoding="utf-8"))
    images = d.get("images", [])
    anns = d.get("annotations", [])
    per_image = collections.Counter(a["image_id"] for a in anns)
    unanswerable = sum(1 for a in anns if not a.get("caption", "").strip())
    text_flagged = sum(1 for i in images if i.get("text_detected"))
    with_q = sum(1 for i in images if i.get("question"))
    cap[split] = {
        "images": len(images),
        "annotations": len(anns),
        "captions_per_image_mean": round(len(anns) / len(images), 2) if images else 0,
        "images_with_10_captions": sum(1 for v in per_image.values() if v >= 10),
        "empty_or_unanswerable_captions": unanswerable,
        "images_flagged_text_detected": text_flagged,
        "images_with_spoken_question": with_q,
        "top_keys": sorted(images[0].keys()) if images else [],
        "example": images[0] if images else None,
    }
OUT["vizwiz_captions"] = cap

# ------------------------------------------------------------- VizWiz-QualityIssues
FLAWS = ["blur", "too_dark", "too_bright", "rotated", "obstructed",
         "poor_framing", "no_flaw", "other"]
qual: dict[str, object] = {}
for split in ("train", "val", "test"):
    fp = BASE / "data/raw/vizwiz/annotations/quality" / f"{split}.json"
    if not fp.exists():
        continue
    rows = json.loads(fp.read_text(encoding="utf-8"))
    flaw_counts = collections.Counter()
    votes_hist = collections.Counter()
    unrecognizable = 0
    for r in rows:
        flaws = r.get("flaws", {}) or {}
        for f in FLAWS:
            if flaws.get(f):
                flaw_counts[f] += 1
        n_votes = sum(1 for k, v in flaws.items() if k != "no_flaw" and v)
        votes_hist[n_votes] += 1
        if r.get("unrecognizable"):
            unrecognizable += 1
    qual[split] = {
        "records": len(rows),
        "unrecognizable": unrecognizable,
        "unrecognizable_rate": round(unrecognizable / len(rows), 3) if rows else 0,
        "flaw_positive_counts": {f: flaw_counts[f] for f in FLAWS},
        "distinct_flaws_per_image_hist": dict(sorted(votes_hist.items())),
        "example": rows[0],
    }
OUT["vizwiz_quality"] = qual

# ---------------------------------------------------------------------- manifests
man: dict[str, object] = {}
mdir = BASE / "data/manifests"
if mdir.is_dir():
    for fp in sorted(mdir.glob("*.jsonl")):
        n = 0
        images_in_manifest = 0
        texts = 0
        with fp.open(encoding="utf-8") as fh:
            first = None
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if first is None:
                    first = json.loads(line)
                if n <= 20000:
                    pass
                images_in_manifest += 1
        man[fp.name] = {"rows": n, "keys": sorted(first.keys()) if first else [],
                        "example": first}
OUT["manifests"] = man

# ------------------------------------------------------------------- image counts
OUT["images_on_disk"] = {
    "flickr30k_images": count_dir(BASE / "data/raw/flickr30k/images"),
    "vizwiz_train_images": count_dir(BASE / "data/raw/vizwiz/images/train"),
    "vizwiz_val_images": count_dir(BASE / "data/raw/vizwiz/images/val"),
    "vizwiz_test_images": count_dir(BASE / "data/raw/vizwiz/images/test"),
}
OUT["bytes_on_disk_gb"] = {
    "flickr30k": dir_bytes(BASE / "data/raw/flickr30k"),
    "vizwiz_images": dir_bytes(BASE / "data/raw/vizwiz/images"),
    "vizwiz_archives": dir_bytes(BASE / "data/raw/vizwiz/archives"),
}

# ------------------------------------------------------------------- feature index
fi = BASE / "data/features"
feat = {}
if fi.is_dir():
    for fp in sorted(fi.glob("*.jsonl")):
        n = 0
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        feat[fp.name] = {"rows": n, "gb": round(fp.stat().st_size / 1e9, 2)}
OUT["feature_indexes"] = feat

# --------------------------------------------------------------- eval run metrics
runs = {}
for fp in sorted((BASE / "runs").rglob("*.json")):
    rel = str(fp.relative_to(BASE))
    try:
        runs[rel] = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostics only
        runs[rel] = f"<unreadable: {exc}>"
OUT["runs"] = runs

print(json.dumps(OUT, indent=1, default=str)[:200000])
