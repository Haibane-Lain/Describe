"""Evaluate a model split by whether the photograph was usable.

The degradation experiment's acceptance criterion is two-sided, and ordinary
evaluation cannot express it:

* the **unrecognizable** bucket must improve from its baseline, **and**
* the **recognizable** bucket must not regress.

Reporting a single aggregate number would let a model win by trading one for the
other, which is exactly the failure mode the criterion exists to prevent.

Baselines to beat (E3, full text decoder, base model):

    recognizable    BLEU-4 0.2089   4-gram 40.7%
    unrecognizable  BLEU-4 0.0500   4-gram  7.6%

The 208 images with no usable caption at all carry no references, so they are
reported separately using reference-free statistics plus the hedging rate.

    py -3.12 -m train.finetune.eval_by_quality \
        --predictions runs/blip_base_degraded/predictions_full_beam5.jsonl \
        --label "base + degradation"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.hedging_analysis import HEDGE  # noqa: E402
from train.finetune.evaluate_blip import collapse_diagnostics  # noqa: E402

# From docs/status-report.md and the E3 run: the numbers a degraded model must beat.
BASELINE = {
    "recognizable": {"BLEU-4": 0.2089, "fourgram": 0.407},
    "unrecognizable": {"BLEU-4": 0.0500, "fourgram": 0.076},
}

# Minimum movement that counts as real, in BLEU-4.
#
# A bare `delta > 0` test is worthless here: the baseline constant is rounded to
# four places, so E3 scored against its own baseline gives delta = +2.9e-5 and
# would "pass" the improvement test. Requiring a margin of half a BLEU-4 point
# - 10% of the unrecognizable baseline, and comfortably above run-to-run noise -
# makes the gate mean something. The no-regression side uses the same margin.
IMPROVEMENT_MARGIN = 0.005
REGRESSION_TOLERANCE = 0.005


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_quality(manifest: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("quality"):
                    out[row["file_name"]] = row["quality"]
    return out


def score(subset: list[dict]) -> dict:
    from train.train.evaluate import score_predictions

    scorable = [p for p in subset if p.get("references")]
    if not scorable:
        return {"n": 0}
    scored = score_predictions(scorable, with_spice=False, with_meteor=False)
    core = scored.get("metrics", {})
    diagnostics = collapse_diagnostics(scorable)
    return {
        "n": len(scorable),
        "BLEU-1": core.get("BLEU-1"),
        "BLEU-4": core.get("BLEU-4"),
        "CIDEr-D": core.get("CIDEr-D"),
        "ROUGE-L": core.get("ROUGE-L"),
        "fourgram_match_rate": diagnostics["fourgram_match_rate"],
        "mean_length": diagnostics["mean_length"],
        "distinct_tokens": diagnostics["distinct_tokens"],
        "empty": diagnostics["empty_predictions"],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Metrics split by image recognizability.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/vizwiz_val.jsonl"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--captionless", type=Path, default=None,
                        help="predictions for the 208 no-usable-caption images, if available")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    predictions = load_predictions(args.predictions)
    quality = load_quality(args.manifest)
    label = args.label or args.predictions.stem

    buckets: dict[str, list[dict]] = {"all": predictions, "recognizable": [], "unrecognizable": []}
    for p in predictions:
        q = quality.get(p.get("file_name")) or {}
        if q.get("unrecognizable"):
            buckets["unrecognizable"].append(p)
        else:
            buckets["recognizable"].append(p)

    print(f"  {label}  ({len(predictions):,} predictions)\n")
    print(f"  {'bucket':<18} {'n':>5} {'BLEU-1':>7} {'BLEU-4':>7} {'CIDEr':>7} "
          f"{'4gram':>7} {'len':>5} {'distinct':>9}")
    results = {}
    for name, subset in buckets.items():
        s = score(subset)
        results[name] = s
        if not s.get("n"):
            continue
        print(f"  {name:<18} {s['n']:>5} {s['BLEU-1']:>7.4f} {s['BLEU-4']:>7.4f} "
              f"{s['CIDEr-D']:>7.4f} {100 * s['fourgram_match_rate']:>6.1f}% "
              f"{s['mean_length']:>5.1f} {s['distinct_tokens']:>9}")

    # The acceptance gate.
    print(f"\n  acceptance gate (vs E3 baseline; margin {IMPROVEMENT_MARGIN:+.4f} BLEU-4)")
    verdict = True
    for name in ("recognizable", "unrecognizable"):
        s = results.get(name) or {}
        if not s.get("n"):
            continue
        base = BASELINE[name]["BLEU-4"]
        delta = (s["BLEU-4"] or 0) - base
        if name == "recognizable":
            ok = delta >= -REGRESSION_TOLERANCE
            note = "no regression" if ok else "REGRESSED"
        else:
            ok = delta >= IMPROVEMENT_MARGIN
            note = "improved" if ok else "no meaningful improvement"
        verdict = verdict and ok
        print(f"    {name:<18} BLEU-4 {s['BLEU-4']:.4f} vs {base:.4f} "
              f"({delta:+.4f})  {note}")
    print(f"\n  {'PASS' if verdict else 'FAIL'} two-sided acceptance")

    if args.captionless and args.captionless.exists():
        cl = load_predictions(args.captionless)
        hedged = sum(1 for p in cl if HEDGE.search(p.get("prediction") or ""))
        lengths = [len((p.get("prediction") or "").split()) for p in cl]
        results["captionless_208"] = {
            "n": len(cl),
            "hedge_rate": hedged / max(len(cl), 1),
            "mean_length": sum(lengths) / max(len(lengths), 1),
            "empty": sum(1 for p in cl if not (p.get("prediction") or "").strip()),
        }
        print(f"\n  208 no-usable-caption images: hedge rate "
              f"{100 * hedged / max(len(cl), 1):.1f}%, mean length "
              f"{sum(lengths) / max(len(lengths), 1):.1f}")

    results["label"] = label
    results["predictions_file"] = str(args.predictions)
    results["baseline"] = BASELINE
    results["acceptance_pass"] = verdict
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
