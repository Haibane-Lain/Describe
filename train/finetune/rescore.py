"""Re-score existing predictions on the selection-free reported split.

Why this exists: per-epoch checkpoint selection used a 1,500-image subset that was
**contained inside** the 7,542 images previously reported on, so every published
number was optimistically biased by selection. The fix is to report only on the
6,042 images that were never used to choose a checkpoint.

Re-scoring is CPU-only and needs no GPU and no re-inference: the prediction files
already exist, and both the captioning metrics and the VizWiz-VQA answer-coverage
metric are computed from text.

    py -3.12 -m train.finetune.rescore --all
    py -3.12 -m train.finetune.rescore \
        --predictions runs/blip_large_head/predictions_full_beam5.jsonl --label "BLIP-large"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune import holdout  # noqa: E402
from train.finetune.eval_vqa import load_vqa, evaluate as evaluate_vqa  # noqa: E402
from train.finetune.evaluate_blip import collapse_diagnostics  # noqa: E402

# The six runs with full-split predictions, best first. Labels are what appear in
# the README tables.
KNOWN_RUNS: list[tuple[str, str]] = [
    ("blip_large_head", "BLIP-large full decoder"),
    ("blip_base_head", "E3 full decoder (base)"),
    ("blip_base_lora", "E1 LoRA"),
    ("blip_base_lora_wide", "E2 LoRA +dense"),
    ("blip_base_textaware", "E4 text-aware"),
    ("blip_zeroshot", "zero-shot BLIP-base"),
]


def load_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def caption_metrics(predictions: list[dict]) -> dict:
    """Corpus metrics through the shared scorer."""
    from train.train.evaluate import score_predictions

    scored = score_predictions(predictions, with_spice=False, with_meteor=False)
    return {**scored, "diagnostics": collapse_diagnostics(predictions)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Re-score on the selection-free split.")
    parser.add_argument("--predictions", type=Path, nargs="*", default=None)
    parser.add_argument("--label", nargs="*", default=None)
    parser.add_argument("--all", action="store_true", help="re-score every known run")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--skip-vqa", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    print(f"  {holdout.describe()}\n")

    if args.all:
        pairs = [
            (Path("runs") / run / "predictions_full_beam5.jsonl", label)
            for run, label in KNOWN_RUNS
        ]
    else:
        if not args.predictions:
            raise SystemExit("pass --all, or --predictions with optional --label")
        labels = args.label or [p.stem for p in args.predictions]
        if len(labels) != len(args.predictions):
            raise SystemExit("--label count must match --predictions count")
        pairs = list(zip(args.predictions, labels))

    vqa_annotations = None if args.skip_vqa else load_vqa("val")

    rows_out: list[dict] = []
    print(f"  {'model':<26} {'n':>5} {'BLEU-1':>7} {'BLEU-4':>7} {'CIDEr':>7} "
          f"{'ROUGE':>7} {'4gram':>7} {'cover':>7}")
    for path, label in pairs:
        if not path.exists():
            print(f"  {label:<26}   missing: {path}")
            continue

        predictions = load_predictions(path)
        reported = holdout.filter_predictions(predictions)  # excludes the selection set
        if not reported:
            print(f"  {label:<26}   no reported rows")
            continue

        result = caption_metrics(reported)
        core = result.get("metrics", {})
        diagnostics = result.get("diagnostics", {})

        coverage = None
        if vqa_annotations is not None:
            caption_by_file = {p["file_name"]: p.get("prediction") or "" for p in reported}
            vqa_result = evaluate_vqa(caption_by_file, vqa_annotations)
            coverage = vqa_result["image_coverage_rate"]

        print(
            f"  {label:<26} {len(reported):>5} {core.get('BLEU-1', 0):>7.4f} "
            f"{core.get('BLEU-4', 0):>7.4f} {core.get('CIDEr-D', 0):>7.4f} "
            f"{core.get('ROUGE-L', 0):>7.4f} "
            f"{100 * diagnostics.get('fourgram_match_rate', 0):>6.1f}% "
            + (f"{100 * coverage:>6.1f}%" if coverage is not None else f"{'-':>7}")
        )

        record = {
            "label": label,
            "predictions_file": str(path),
            "split": "reported (selection-free)",
            "n_images": len(reported),
            "selection_excluded": len(predictions) - len(reported),
            "metrics": core,
            "diagnostics": diagnostics,
            "vqa_answer_coverage": coverage,
        }
        rows_out.append(record)

        out_path = args.out_dir / path.parent.name / "val_reported.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    summary = args.out_dir / "reported_split_summary.json"
    summary.write_text(
        json.dumps(
            {
                "split": "reported (selection-free)",
                "selection_size": holdout.SELECTION_SIZE,
                "selection_seed": holdout.SELECTION_SEED,
                "note": (
                    "All figures computed on validation images that were NOT used for "
                    "per-epoch checkpoint selection. Replaces any earlier figure computed "
                    "on the full 7,542."
                ),
                "models": rows_out,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        print(f"\n  summary written: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
