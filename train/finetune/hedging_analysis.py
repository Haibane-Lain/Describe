"""Do the captions hedge when the photograph is unusable?

The project's thesis is that the system should prefer an honest admission of
uncertainty over a confident misdescription. If the captioner's language already
tracks image quality - hedging on bad photographs, asserting on good ones - then
part of that behaviour exists and can be measured, and the router's hedging rule
has something real to key on.

Measured over:
  * the 208 validation images with **zero** usable captions (100% of them are
    annotated `unrecognizable` by crowdworkers), and
  * all 7,542 scorable validation images, bucketed by the same annotation.

No references are needed for this, which matters: those 208 images are excluded
from every BLEU-style evaluation because they have no reference captions at all.
They are the hardest images in the set and they were previously unmeasured.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEDGE_PATTERNS = [
    r"appears to be", r"seems to be", r"looks like", r"may be", r"possibly",
    r"blurry", r"unclear", r"hard to tell", r"cannot tell", r"can't tell",
    r"not sure", r"too blurry", r"too dark", r"quality issue", r"severe",
    r"difficult to", r"partially", r"obscured", r"out of focus", r"unsure",
    r"might be", r"something", r"some kind of",
]
HEDGE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# Assertive openers: a caption that commits to a specific object.
ASSERTIVE = re.compile(
    r"^(a|an|the)\s+\w+", re.IGNORECASE
)


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quality_of(path: Path) -> dict[str, dict]:
    out = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("quality"):
                    out[row["file_name"]] = row["quality"]
    return out


def report(label: str, predictions: list[dict]) -> dict:
    n = len(predictions)
    hedged = sum(1 for p in predictions if HEDGE.search(p.get("prediction") or ""))
    lengths = [len((p.get("prediction") or "").split()) for p in predictions]
    empty = sum(1 for p in predictions if not (p.get("prediction") or "").strip())
    print(
        f"  {label:<34} n={n:>5}  hedged {100 * hedged / max(n, 1):>5.1f}%  "
        f"mean len {sum(lengths) / max(len(lengths), 1):>5.1f}  empty {empty}"
    )
    return {
        "n": n,
        "hedge_rate": hedged / max(n, 1),
        "mean_length": sum(lengths) / max(len(lengths), 1),
        "empty": empty,
    }


def main() -> int:
    root = Path(".")
    all_preds = load_predictions(root / "runs/blip_large_head/predictions_full_beam5.jsonl")
    quality = quality_of(root / "data/manifests/vizwiz_val.jsonl")

    captionless = load_predictions(root / "runs/blip_large_head/captionless208.jsonl")

    print("Hedging vs image quality (BLIP-large)\n")
    results = {}
    results["captionless_208"] = report("208 images, no usable caption", captionless)

    for flag, name in ((True, "unrecognizable (crowdworker)"), (False, "recognizable")):
        subset = [
            p for p in all_preds
            if (quality.get(p.get("file_name")) or {}).get("unrecognizable") is flag
        ]
        results["unrecognizable" if flag else "recognizable"] = report(name, subset)

    # by dominant flaw
    def dominant(quality_row: dict) -> str | None:
        flaws = (quality_row or {}).get("flaws") or {}
        ranked = sorted(flaws.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] == 0:
            return None
        return ranked[0][0]

    print()
    for code, name in (("BLR", "blur"), ("DRK", "too dark"), ("BRT", "too bright"),
                       ("FRM", "poor framing"), ("ROT", "rotated")):
        subset = [p for p in all_preds if dominant(quality.get(p.get("file_name"))) == code]
        if len(subset) >= 20:
            results[f"flaw_{name}"] = report(f"dominant flaw: {name}", subset)

    print()
    hl = results["captionless_208"]["hedge_rate"]
    hr = results["recognizable"]["hedge_rate"]
    hu = results["unrecognizable"]["hedge_rate"]
    print(f"  hedge rate, no-usable-caption images : {100 * hl:.1f}%")
    print(f"  hedge rate, unrecognizable           : {100 * hu:.1f}%")
    print(f"  hedge rate, recognizable             : {100 * hr:.1f}%")
    print(f"  ratio (unrecognizable / recognizable): {hu / max(hr, 1e-9):.2f}x")

    out = root / "runs/blip_large_head/hedging_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
