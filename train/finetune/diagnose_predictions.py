"""Re-derive collapse diagnostics and (optionally) split metrics for any prediction log.

Why this exists as a separate tool: the from-scratch captioner and the
fine-tuned BLIP must be judged by **the same code**, or the comparison in the
write-up is between two different measurements rather than between two models.
The from-scratch run's prediction logs already exist on disk, so re-scoring them
through this tool costs seconds and removes any excuse for comparing BLEU-4
computed two different ways.

The diagnostics are the point. BLEU-4 alone did not reveal that the from-scratch
model was collapsed; distinct-token count, top-token share, and 4-gram match
rate did (see ``docs/status-report.md``).

    # diagnostics for an existing run
    py -3.12 -m train.finetune.diagnose_predictions runs/vizwiz_ft2/preds_vizwiz_val.jsonl

    # with the quality/flaw breakdown joined from the manifest
    py -3.12 -m train.finetune.diagnose_predictions runs/blip_vizwiz_lora/predictions_full.jsonl \
        --manifest data/manifests/vizwiz_val.jsonl --out runs/blip_vizwiz_lora/diagnostics.json

The prediction JSONL format is the one ``evaluate.py`` and
``evaluate_blip.py`` both write: ``image_id``, ``file_name``, ``text_detected``,
``prediction``, ``references``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.evaluate_blip import collapse_diagnostics  # noqa: E402

FLAW_NAMES = {
    "BLR": "blur",
    "BRT": "too bright",
    "DRK": "too dark",
    "FRM": "poor framing",
    "NON": "no flaw",
    "OBS": "obstructed",
    "OTH": "other",
    "ROT": "rotated",
}


def read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_quality(manifest: Path) -> dict[str, dict]:
    """``file_name -> quality annotation`` from a VizWiz manifest."""
    out: dict[str, dict] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("quality"):
                out[row["file_name"]] = row["quality"]
    return out


def dominant_flaw(quality: dict) -> str | None:
    flaws = (quality or {}).get("flaws") or {}
    ranked = sorted(flaws.items(), key=lambda kv: -kv[1])
    if not ranked or ranked[0][1] == 0:
        return None
    return ranked[0][0]


def summarise(predictions: list[dict], label: str) -> dict:
    """Diagnostics plus a few length/no-output statistics."""
    diagnostics = collapse_diagnostics(predictions)
    references = [p for p in predictions if p.get("references")]
    diagnostics["label"] = label
    diagnostics["n_with_references"] = len(references)
    diagnostics["reference_vocab_note"] = (
        "distinct_tokens is a fraction of the model's output vocabulary; "
        "reference_vocab_size is the vocabulary the human references use"
    )
    return diagnostics


def split_metrics(
    predictions: list[dict],
    quality_by_name: dict[str, dict],
    *,
    with_meteor: bool = False,
) -> dict:
    """Metric buckets by unrecognizability and by dominant flaw.

    Joined from the manifest's own ``quality`` field rather than from a gate
    re-run: the annotation is what the gate is *trained to predict*, so using it
    directly measures the thing the report is about - how captioning degrades as
    the photograph gets worse - without adding the gate's own error on top.
    """
    from train.train.evaluate import score_predictions

    def bucket(label: str, predicate) -> dict | None:
        subset = [p for p in predictions if p.get("references") and predicate(p)]
        if len(subset) < 20:
            return None
        try:
            scored = score_predictions(subset, with_spice=False, with_meteor=with_meteor)
        except Exception:  # noqa: BLE001 - a thin bucket must not kill the report
            return None
        core = scored.get("metrics", {})
        diagnostic = collapse_diagnostics(subset)
        return {
            "n": len(subset),
            "BLEU-1": core.get("BLEU-1"),
            "BLEU-4": core.get("BLEU-4"),
            "CIDEr-D": core.get("CIDEr-D"),
            "ROUGE-L": core.get("ROUGE-L"),
            "mean_length": diagnostic["mean_length"],
            "fourgram_match_rate": diagnostic["fourgram_match_rate"],
            "distinct_tokens": diagnostic["distinct_tokens"],
            "empty_predictions": diagnostic["empty_predictions"],
        }

    def unrecognizable(p) -> bool:
        return bool((quality_by_name.get(p.get("file_name")) or {}).get("unrecognizable"))

    def flawed(p) -> bool:
        quality = quality_by_name.get(p.get("file_name")) or {}
        flaw = dominant_flaw(quality)
        return flaw not in (None, "NON")

    out: dict[str, dict] = {}
    for label, predicate in (
        ("all", lambda p: True),
        ("unrecognizable_true", unrecognizable),
        ("unrecognizable_false", lambda p: not unrecognizable(p) and p.get("file_name") in quality_by_name),
        ("has_dominant_flaw", flawed),
        ("no_flaw", lambda p: dominant_flaw(quality_by_name.get(p.get("file_name")) or {}) in (None, "NON")
                              and p.get("file_name") in quality_by_name),
        ("text_detected_true", lambda p: p.get("text_detected") is True),
        ("text_detected_false", lambda p: p.get("text_detected") is False),
    ):
        result = bucket(label, predicate)
        if result is not None:
            out[label] = result

    per_flaw: dict[str, dict] = {}
    for code, name in FLAW_NAMES.items():
        if code == "NON":
            continue
        result = bucket(
            f"flaw_{name}",
            lambda p, c=code: dominant_flaw(quality_by_name.get(p.get("file_name")) or {}) == c,
        )
        if result is not None:
            per_flaw[name] = result
    if per_flaw:
        out["by_dominant_flaw"] = per_flaw

    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collapse diagnostics for a prediction log.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="VizWiz manifest supplying quality annotations for the splits")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--with-meteor", action="store_true")
    args = parser.parse_args(argv)

    predictions = read_predictions(args.predictions)
    label = args.label or args.predictions.stem
    result = summarise(predictions, label)

    print(f"  {label}: {len(predictions)} predictions, "
          f"{result['n_with_references']} with references")
    print(f"    distinct tokens      {result['distinct_tokens']}  "
          f"(references use {result['reference_vocab_size']})")
    print(f"    top token            {result['top_token']!r} at {100 * result['top_token_share']:.1f}%")
    print(f"    top-10 share         {100 * result['top10_share']:.1f}%")
    print(f"    bigram match rate    {100 * result['bigram_match_rate']:.1f}%")
    print(f"    4-gram match rate    {100 * result['fourgram_match_rate']:.1f}%")
    print(f"    mean length          {result['mean_length']:.1f} tokens")
    print(f"    empty predictions    {result['empty_predictions']}")

    if args.manifest is not None:
        quality = read_quality(args.manifest)
        covered = sum(1 for p in predictions if p.get("file_name") in quality)
        print(f"\n  quality join: {covered}/{len(predictions)} predictions matched the manifest")
        splits = split_metrics(predictions, quality, with_meteor=args.with_meteor)
        result["splits"] = splits

        print(f"\n  {'bucket':<24} {'n':>6} {'BLEU-1':>8} {'BLEU-4':>8} {'CIDEr-D':>8} "
              f"{'mean len':>9} {'4-gram':>8}")
        for name, values in splits.items():
            if name == "by_dominant_flaw":
                continue
            print(f"  {name:<24} {values['n']:>6} {values['BLEU-1']:>8.4f} {values['BLEU-4']:>8.4f} "
                  f"{values['CIDEr-D']:>8.4f} {values['mean_length']:>9.1f} "
                  f"{100 * values['fourgram_match_rate']:>7.1f}%")
        if "by_dominant_flaw" in splits:
            print(f"\n  worst-photograph breakdown (dominant flaw per image)")
            for name, values in sorted(splits["by_dominant_flaw"].items(), key=lambda kv: kv[1]["BLEU-4"] or 0):
                print(f"  {name:<24} {values['n']:>6} {values['BLEU-1']:>8.4f} {values['BLEU-4']:>8.4f} "
                      f"{values['CIDEr-D']:>8.4f} {values['mean_length']:>9.1f} "
                      f"{100 * values['fourgram_match_rate']:>7.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
