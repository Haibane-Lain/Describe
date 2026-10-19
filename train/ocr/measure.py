"""Phase 0 and Phase 3: what does OCR actually cover?

``train/docs/ocr-plan.md`` §3 asks for one number before any accuracy target:
run OCR on the same validation images the captioner was measured on, and ask
whether the extracted text contains the crowdworker majority answer. That is
this tool. It also answers §6 Phase 3 -- is the answer in the caption **or** the
OCR output -- which is the honest combined coverage figure.

Three rules are enforced here rather than left to discipline:

* **One scoring rule.** Hits are ``train.finetune.eval_vqa.answer_covered``, the
  function behind the published 11.7% captioner number and the consistent intent
  table. ``reading_gap.answer_in_caption`` is a near-duplicate with a slightly
  different tokeniser; using it would recreate the two-scoring-rule problem that
  already produced 2,923 against 3,048 in this project.
* **Every rate names its population.** A = 577 (all answerable reading
  questions), B = 575 (the subset with a captioner prediction). See
  ``train.ocr.reading_set`` for why the two differ.
* **The 577 is not for tuning.** Engine-parameter overrides are refused on
  ``--split val`` unless a frozen ``--config-file`` is supplied, so a config that
  was selected by looking at the test set cannot be passed off as a measurement.
  Sweeps belong on ``--split train`` (1,980 questions).

    py -3.12 -m train.ocr.measure --split val --pipelines none,enhance
    py -3.12 -m train.ocr.measure --split val \
        --combined runs/blip_large_head/predictions_full_beam5.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.eval_vqa import (  # noqa: E402
    answer_covered,
    content_terms,
    load_predictions,
    load_vqa,
    majority_answer,
    tokenize,
)
from train.finetune.intent_table import classify as classify_intent  # noqa: E402
from train.finetune.reading_gap import classify_kind, is_reading_question  # noqa: E402
from train.ocr.engine import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_CANVAS_SIZE,
    DEFAULT_LINK_THRESHOLD,
    DEFAULT_LOW_TEXT,
    DEFAULT_MAG_RATIO,
    DEFAULT_TEXT_THRESHOLD,
    LOW_CONFIDENCE,
    MODEL_DIR,
    EasyOcrBackend,
    OcrEngine,
)
from train.ocr.preprocess import PreprocessConfig  # noqa: E402
from train.ocr.reading_set import (  # noqa: E402
    MIN_REPORTABLE_N,
    reading_items,
    reading_set_stats,
)
from train.ocr.word_metrics import cer  # noqa: E402

DEFAULT_OUT_DIR = Path("runs/ocr_easyocr")
DEFAULT_CAPTIONS = Path("runs/blip_large_head/predictions_full_beam5.jsonl")

# The preprocessing variants the measurement compares. "none" is the honest
# baseline -- EasyOCR on the photograph as it was taken. "enhance" adds the
# geometry steps build-plan 2.4 promises; "enhance+contrast" adds CLAHE, which is
# off by default because EasyOCR already retries low-confidence crops with its own
# contrast adjustment.
PIPELINES: dict[str, PreprocessConfig] = {
    "none": PreprocessConfig(upscale=False, deskew=False, perspective=False, enhance=False),
    "enhance": PreprocessConfig(),
    "enhance+contrast": PreprocessConfig(enhance=True),
}


@dataclass(frozen=True)
class Measured:
    """One question, its answer, and the two classifications that matter."""

    file_name: str
    path: Path
    question: str
    answer: str
    intent: str
    subtype: str | None


def answer_hit(text: str, answer: str) -> bool:
    """The single scoring rule. Imported so no caller can drift from it."""
    return answer_covered(text, answer)


def rate(hits: int, n: int) -> float:
    return hits / n if n else 0.0


def reading_questions(split: str) -> list[Measured]:
    """Population A: every answerable reading question in ``split``."""
    return [
        Measured(
            file_name=item.file_name,
            path=item.path,
            question=item.question,
            answer=item.answer,
            intent="reading / detail",
            subtype=item.subtype,
        )
        for item in reading_items(split)
    ]


def all_questions(split: str) -> list[Measured]:
    """Every answerable question in ``split``, all intents (for ``--all-val``)."""
    images_root = Path("data/raw/vizwiz/images") / split
    measured: list[Measured] = []
    missing: list[str] = []
    for row in load_vqa(split):
        if int(row.get("answerable", 0)) != 1:
            continue
        answer = majority_answer(row)
        if not answer:
            continue
        file_name = row["image"]
        path = images_root / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        question = row.get("question") or ""
        intent = classify_intent(question)
        measured.append(
            Measured(
                file_name=file_name,
                path=path,
                question=question,
                answer=answer,
                intent=intent,
                subtype=classify_kind(question) if intent == "reading / detail" else None,
            )
        )
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) missing under {images_root}; refusing to measure."
        )
    measured.sort(key=lambda item: item.file_name)
    return measured


def manifest_signals(split: str) -> dict[str, dict[str, Any]]:
    """Free per-image signals from the manifest: text present, gate verdict, flaws.

    No model is run for these. ``text_detected`` comes from the VizWiz caption
    annotations and ``quality.*`` from the quality annotations, which is exactly
    the "coordinate so OCR is not blamed for unusable images" check that
    ``ocr-plan.md`` §8 asks for.
    """
    path = Path(f"data/manifests/vizwiz_{split}.jsonl")
    signals: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return signals
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            quality = row.get("quality") or {}
            signals[row.get("file_name")] = {
                "text_detected": row.get("text_detected"),
                "unrecognizable": quality.get("unrecognizable"),
            }
    return signals


def _load_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[(row["file_name"], row["pipeline"])] = row
    return rows


def _breakdown(
    items: list[Measured],
    texts: dict[str, str],
    captions: dict[str, str] | None,
    keys: dict[str, str],
    key_name: str,
) -> list[dict[str, Any]]:
    """Per-group OCR / caption / either / both counts, with n on every figure."""
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n_a": 0, "ocr_a": 0, "n_b": 0, "ocr_b": 0, "caption_b": 0, "either_b": 0, "both_b": 0}
    )
    for item in items:
        bucket = buckets[keys[item.file_name]]
        text = texts.get(item.file_name, "")
        hit = answer_hit(text, item.answer)
        bucket["n_a"] += 1
        bucket["ocr_a"] += int(hit)
        if captions is not None and item.file_name in captions:
            caption_hit = answer_hit(captions[item.file_name], item.answer)
            bucket["n_b"] += 1
            bucket["ocr_b"] += int(hit)
            bucket["caption_b"] += int(caption_hit)
            bucket["either_b"] += int(hit or caption_hit)
            bucket["both_b"] += int(hit and caption_hit)

    rows: list[dict[str, Any]] = []
    for name, counts in sorted(buckets.items(), key=lambda kv: -kv[1]["n_a"]):
        row: dict[str, Any] = {
            key_name: name,
            "n": counts["n_a"],
            "ocr_hit": counts["ocr_a"],
            "ocr_rate": rate(counts["ocr_a"], counts["n_a"]),
            "publish": counts["n_a"] >= MIN_REPORTABLE_N,
        }
        if captions is not None:
            row.update(
                {
                    "n_with_caption": counts["n_b"],
                    "ocr_hit_with_caption": counts["ocr_b"],
                    "caption_hit": counts["caption_b"],
                    "caption_rate": rate(counts["caption_b"], counts["n_b"]),
                    "either_hit": counts["either_b"],
                    "either_rate": rate(counts["either_b"], counts["n_b"]),
                    "both_hit": counts["both_b"],
                    "both_rate": rate(counts["both_b"], counts["n_b"]),
                }
            )
        rows.append(row)
    return rows


def _totals(rows: list[dict[str, Any]], captions: dict[str, str] | None) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    total_hit = sum(row["ocr_hit"] for row in rows)
    summary: dict[str, Any] = {
        "n": total_n,
        "ocr_hit": total_hit,
        "ocr_rate": rate(total_hit, total_n),
    }
    if captions is not None:
        n_b = sum(row.get("n_with_caption", 0) for row in rows)
        summary.update(
            {
                "n_with_caption": n_b,
                "ocr_hit_with_caption": sum(row.get("ocr_hit_with_caption", 0) for row in rows),
                "caption_hit": sum(row.get("caption_hit", 0) for row in rows),
                "either_hit": sum(row.get("either_hit", 0) for row in rows),
                "both_hit": sum(row.get("both_hit", 0) for row in rows),
            }
        )
        summary["caption_rate"] = rate(summary["caption_hit"], n_b)
        summary["either_rate"] = rate(summary["either_hit"], n_b)
        summary["both_rate"] = rate(summary["both_hit"], n_b)
        summary["ocr_rate_with_caption"] = rate(summary["ocr_hit_with_caption"], n_b)
    return summary


def near_miss_analysis(items: list[Measured], texts: dict[str, str]) -> dict[str, Any]:
    """How close were the misses -- detection failure, or one character off?

    The headline hit rate cannot distinguish "OCR found nothing" from "OCR read
    ``TortiLA`` where the answer was ``tortilla``". Those imply opposite next
    steps (better detection versus better recognition), so the misses are scored
    by the character error rate of the closest same-length window of recognised
    tokens against the answer phrase. ``ocr-plan.md`` §4 names character-level
    accuracy on dense small text as the core difficulty; this is the measurement
    that says whether that is where the losses actually are.
    """
    buckets = {
        "hit": 0,
        "no_text": 0,
        "within_0.1": 0,
        "within_0.25": 0,
        "within_0.5": 0,
        "beyond_0.5": 0,
    }
    examples: list[dict[str, Any]] = []

    for item in items:
        text = texts.get(item.file_name, "")
        if answer_hit(text, item.answer):
            buckets["hit"] += 1
            continue
        tokens = tokenize(text)
        if not tokens:
            buckets["no_text"] += 1
            continue

        terms = content_terms(item.answer)
        width = max(1, len(terms))
        best = 1.0
        best_window = ""
        for start in range(0, max(1, len(tokens) - width + 1)):
            window = " ".join(tokens[start : start + width])
            distance = cer(window, item.answer)
            if distance < best:
                best = distance
                best_window = window

        if best <= 0.1:
            buckets["within_0.1"] += 1
        elif best <= 0.25:
            buckets["within_0.25"] += 1
        elif best <= 0.5:
            buckets["within_0.5"] += 1
        else:
            buckets["beyond_0.5"] += 1

        if best <= 0.25 and len(examples) < 12:
            examples.append(
                {
                    "file_name": item.file_name,
                    "subtype": item.subtype,
                    "answer": item.answer,
                    "closest_window": best_window,
                    "cer": round(best, 4),
                }
            )

    misses = len(items) - buckets["hit"]
    buckets["misses"] = misses
    buckets["near_miss_share"] = (
        (buckets["within_0.1"] + buckets["within_0.25"] + buckets["within_0.5"]) / misses
        if misses
        else 0.0
    )
    buckets["examples"] = examples
    return buckets


def _print_rows(rows: list[dict[str, Any]], label: str, captions: bool) -> None:
    header = f"  {label:<26} {'n':>5} {'ocr':>5} {'rate':>7}"
    if captions:
        header += f" {'cap':>5} {'either':>7}"
    print(header)
    for row in rows:
        if row["n"] < MIN_REPORTABLE_N:
            marker = "  (n<20 -- do not publish)"
        else:
            marker = ""
        line = (
            f"  {row[label]:<26} {row['n']:>5} {row['ocr_hit']:>5} "
            f"{100 * row['ocr_rate']:>6.1f}%"
        )
        if captions:
            line += (
                f" {100 * row.get('caption_rate', 0.0):>5.1f}%"
                f" {100 * row.get('either_rate', 0.0):>6.1f}%"
            )
        print(line + marker)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure OCR answer coverage on VizWiz reading questions."
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--all-val", action="store_true",
                        help="OCR every answerable val question, not just reading "
                             "(4319 images; gives the full intent table its OCR column)")
    parser.add_argument("--pipelines", default="none,enhance",
                        help=f"comma-separated preprocess variants: {', '.join(PIPELINES)}")
    parser.add_argument("--combined", type=Path, default=None,
                        help="captioner predictions JSONL for the caption-or-OCR number")
    parser.add_argument("--config-file", type=Path, default=None,
                        help="frozen engine config (required for any override on val)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--predictions-pipeline", default=None,
                        help="pipeline whose text is written as OCR predictions "
                             "(default: the first one requested)")
    parser.add_argument("--resume", action="store_true", help="skip rows already on disk")
    parser.add_argument("--overwrite", action="store_true", help="replace existing rows")
    parser.add_argument("--partial", action="store_true",
                        help="write the report even if some rows are missing")
    parser.add_argument("--limit", type=int, default=None, help="measure only the first N")
    parser.add_argument("--gpu", dest="gpu", action="store_true", default=None)
    parser.add_argument("--cpu", dest="gpu", action="store_false")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--canvas-size", type=int, default=None)
    parser.add_argument("--mag-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--low-confidence", type=float, default=LOW_CONFIDENCE)
    args = parser.parse_args(argv)

    requested = [name.strip() for name in args.pipelines.split(",") if name.strip()]
    unknown = [name for name in requested if name not in PIPELINES]
    if unknown:
        raise SystemExit(
            f"unknown pipeline(s) {unknown}; choose from {sorted(PIPELINES)}"
        )
    pipeline_names = requested

    overrides = {
        "canvas_size": args.canvas_size,
        "mag_ratio": args.mag_ratio,
        "batch_size": args.batch_size,
    }
    has_overrides = any(value is not None for value in overrides.values())
    frozen: dict[str, Any] = {}
    if args.config_file is not None:
        frozen = json.loads(args.config_file.read_text(encoding="utf-8"))
    elif has_overrides and args.split == "val":
        raise SystemExit(
            "Refusing to override engine parameters on --split val.\n"
            "  The 577 validation images are the only clean measurement of the reading\n"
            "  bucket (ocr-plan.md §2). Sweep on --split train, freeze the chosen values\n"
            "  into a JSON file, and pass it with --config-file.\n"
            "  Offending values: "
            + ", ".join(f"{k}={v}" for k, v in overrides.items() if v is not None)
        )

    engine_params = {
        "canvas_size": args.canvas_size or frozen.get("canvas_size", DEFAULT_CANVAS_SIZE),
        "mag_ratio": args.mag_ratio if args.mag_ratio is not None else frozen.get("mag_ratio", DEFAULT_MAG_RATIO),
        "batch_size": args.batch_size or frozen.get("batch_size", DEFAULT_BATCH_SIZE),
        "text_threshold": frozen.get("text_threshold", DEFAULT_TEXT_THRESHOLD),
        "link_threshold": frozen.get("link_threshold", DEFAULT_LINK_THRESHOLD),
        "low_text": frozen.get("low_text", DEFAULT_LOW_TEXT),
    }
    if "preprocess" in frozen:
        # A frozen file may pin one explicit preprocessing configuration, in which
        # case every requested pipeline name measures that same config. That is how
        # a tuned setup -- not necessarily "none" or "enhance" -- is re-measured.
        pinned_config = PreprocessConfig(**frozen["preprocess"])
        pipeline_configs = {name: pinned_config for name in pipeline_names}
    else:
        pipeline_configs = {name: PIPELINES[name] for name in pipeline_names}

    items = all_questions(args.split) if args.all_val else reading_questions(args.split)
    if args.limit:
        items = items[: args.limit]

    captions: dict[str, str] | None = None
    if args.combined is not None:
        captions = load_predictions(args.combined)

    signals = manifest_signals(args.split)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all" if args.all_val else "reading"
    regions_path = args.out_dir / f"regions_{args.split}_{suffix}.jsonl"
    predictions_path = args.out_dir / f"predictions_{args.split}_{suffix}.jsonl"
    measure_path = args.out_dir / f"measure_{args.split}_{suffix}.json"
    engine_path = args.out_dir / "engine.json"

    if args.overwrite and regions_path.exists():
        regions_path.unlink()
    existing = {} if args.overwrite else _load_rows(regions_path)

    planned = [(item, name) for item in items for name in pipeline_names]
    pending = [
        (item, name) for item, name in planned
        if (item.file_name, name) not in existing
    ]

    print(f"  split        : {args.split}")
    print(f"  questions    : {len(items):,} ({suffix})")
    print(f"  pipelines    : {', '.join(pipeline_names)}")
    print(f"  images to run: {len(pending):,} of {len(planned):,}")
    if captions is not None:
        print(f"  captions     : {args.combined} ({len(captions):,} rows)")
    print()

    started = time.perf_counter()
    backend = None
    if pending:
        backend = EasyOcrBackend(gpu=args.gpu, model_dir=args.model_dir, **engine_params)
        print(f"  device       : {backend.device}")
        engines = {
            name: OcrEngine(
                backend,
                preprocess=pipeline_configs[name],
                low_confidence_below=args.low_confidence,
                pipeline=name,
            )
            for name in pipeline_names
        }
        with regions_path.open("a", encoding="utf-8") as handle:
            for index, (item, name) in enumerate(pending, start=1):
                result = engines[name].read_text(item.path)
                row = result.as_dict()
                row["file_name"] = item.file_name
                row["pipeline"] = name
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                if index % 25 == 0 or index == len(pending):
                    elapsed = time.perf_counter() - started
                    per_run = elapsed / index
                    remaining = (len(pending) - index) * per_run
                    print(
                        f"    {index}/{len(pending)}  {per_run:.2f} s/run  "
                        f"eta {remaining / 60:.1f} min",
                        flush=True,
                    )
    elapsed = time.perf_counter() - started

    rows = _load_rows(regions_path)
    missing = [
        (item.file_name, name) for item, name in planned
        if (item.file_name, name) not in rows
    ]
    if missing and not args.partial:
        raise SystemExit(
            f"{len(missing)} row(s) missing from {regions_path}. "
            "Re-run with --resume, or pass --partial to report on what exists."
        )

    # The engine description has to survive a --resume regeneration, which runs no
    # OCR at all and therefore constructs no reader. Without this, re-deriving the
    # report from saved rows would drop the device and version provenance.
    if backend is not None:
        engine_info: dict[str, Any] | None = backend.describe()
        engine_info["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        engine_path.write_text(json.dumps(engine_info, indent=2), encoding="utf-8")
    elif engine_path.exists():
        engine_info = json.loads(engine_path.read_text(encoding="utf-8"))
        engine_info["note"] = (
            "recovered from engine.json: this run re-scored saved rows and "
            "constructed no reader"
        )
    else:
        devices = sorted({str(row["meta"].get("device")) for row in rows.values()})
        backends = sorted({str(row["meta"].get("backend")) for row in rows.values()})
        engine_info = {
            "backend": backends[0] if len(backends) == 1 else backends,
            "device": devices[0] if len(devices) == 1 else devices,
            "note": "recovered from saved rows; no reader was constructed this run",
        }

    intent_keys = {item.file_name: item.intent for item in items}
    subtype_keys = {
        item.file_name: (item.subtype or "other") for item in items if item.subtype
    }

    report: dict[str, Any] = {
        "generated_by": "train.ocr.measure",
        "split": args.split,
        "population": suffix,
        "questions": len(items),
        "partial": bool(missing),
        "missing_rows": len(missing),
        "config": {
            "engine_parameters": engine_params,
            "pipelines": {name: config.as_dict() for name, config in pipeline_configs.items()},
            "frozen_config_file": str(args.config_file) if args.config_file else None,
            "frozen_config": frozen or None,
            "low_confidence_below": args.low_confidence,
            "model_dir": str(args.model_dir),
            "scoring": "train.finetune.eval_vqa.answer_covered",
            "populations": {
                "A": "all answerable reading questions",
                "B": "A, restricted to rows with a captioner prediction",
            },
        },
        "engine": engine_info,
        "timing": {
            "seconds": round(elapsed, 1),
            "images_run": len(pending),
            "seconds_per_image": round(elapsed / len(pending), 3) if pending else None,
        },
        "pipelines": {},
    }

    for name in pipeline_names:
        texts = {
            file_name: row["text"]
            for (file_name, pipeline), row in rows.items()
            if pipeline == name
        }
        latencies = [
            row["meta"]["latency_ms"]
            for (file_name, pipeline), row in rows.items()
            if pipeline == name
        ]
        empty = sum(1 for file_name in texts if not texts[file_name].strip())
        low = sum(
            1 for (file_name, pipeline), row in rows.items()
            if pipeline == name and row.get("low_confidence")
        )
        entry: dict[str, Any] = {
            "images": len(texts),
            "empty_text": empty,
            "low_confidence": low,
            "latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        }
        if subtype_keys:
            subtype_rows = _breakdown(items, texts, captions, subtype_keys, "subtype")
            entry["subtypes"] = subtype_rows
            entry["reading_totals"] = _totals(subtype_rows, captions)
        if args.all_val:
            intent_rows = _breakdown(items, texts, captions, intent_keys, "intent")
            entry["intents"] = intent_rows
            entry["intent_totals"] = _totals(intent_rows, captions)
        entry["by_text_detected"] = _signal_split(items, texts, captions, signals, "text_detected")
        entry["by_gate_label"] = _signal_split(items, texts, captions, signals, "unrecognizable")
        entry["near_miss"] = near_miss_analysis(items, texts)
        report["pipelines"][name] = entry

    primary = args.predictions_pipeline or pipeline_names[0]
    primary_texts = {
        file_name: row["text"]
        for (file_name, pipeline), row in rows.items()
        if pipeline == primary
    }
    with predictions_path.open("w", encoding="utf-8") as handle:
        for item in items:
            if item.file_name in primary_texts:
                handle.write(
                    json.dumps(
                        {"file_name": item.file_name, "prediction": primary_texts[item.file_name]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    report["predictions_file"] = str(predictions_path)
    report["regions_file"] = str(regions_path)
    report["reading_set"] = reading_set_stats(reading_items(args.split)) if not args.all_val else None
    measure_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- console summary -------------------------------------------------
    print()
    for name in pipeline_names:
        entry = report["pipelines"][name]
        print(f"  pipeline: {name}  ({entry['images']:,} images, "
              f"{entry['empty_text']:,} with no text, "
              f"{entry['low_confidence']:,} low confidence)")
        if "subtypes" in entry:
            totals = entry["reading_totals"]
            print(f"    population A ({totals['n']:,}): OCR {totals['ocr_hit']:,} = "
                  f"{100 * totals['ocr_rate']:.1f}%")
            if captions is not None:
                print(f"    population B ({totals['n_with_caption']:,}): "
                      f"caption {100 * totals['caption_rate']:.1f}%  "
                      f"OCR {100 * totals['ocr_rate_with_caption']:.1f}%  "
                      f"either {100 * totals['either_rate']:.1f}%  "
                      f"both {100 * totals['both_rate']:.1f}%")
            print()
            _print_rows(entry["subtypes"], "subtype", captions is not None)
        if "intents" in entry:
            print()
            _print_rows(entry["intents"], "intent", captions is not None)
        print()
        near = entry.get("near_miss") or {}
        if near:
            misses = near.get("misses", 0)
            print(f"    misses: {misses:,}  (no text at all {near['no_text']:,}; "
                  f"within 0.1 CER {near['within_0.1']:,}; within 0.25 {near['within_0.25']:,}; "
                  f"within 0.5 {near['within_0.5']:,}; beyond {near['beyond_0.5']:,})")
            print(f"    near-miss share of misses (CER <= 0.5): {100 * near['near_miss_share']:.1f}%")
        print()
        for label, key in (("text_detected", "by_text_detected"), ("gate label", "by_gate_label")):
            blocks = entry.get(key) or []
            if not blocks:
                continue
            print(f"    by {label}:")
            for block in blocks:
                print(f"      {str(block['name']):<10} n={block['n']:<5} "
                      f"OCR {100 * block['ocr_rate']:>5.1f}%"
                      + (f"  caption {100 * block.get('caption_rate', 0):>5.1f}%"
                         if captions is not None else ""))
        print()

    print(f"  report      : {measure_path}")
    print(f"  regions     : {regions_path}")
    print(f"  predictions : {predictions_path}")
    if missing:
        print(f"  WARNING: {len(missing)} row(s) missing; report is partial")
    return 0


def _signal_split(
    items: list[Measured],
    texts: dict[str, str],
    captions: dict[str, str] | None,
    signals: dict[str, dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    """Hit rate split by a manifest signal (text present, gate verdict)."""
    if not signals:
        return []
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "ocr": 0, "caption": 0, "n_b": 0}
    )
    for item in items:
        signal = signals.get(item.file_name, {}).get(field)
        bucket = buckets[str(signal)]
        hit = answer_hit(texts.get(item.file_name, ""), item.answer)
        bucket["n"] += 1
        bucket["ocr"] += int(hit)
        if captions is not None and item.file_name in captions:
            bucket["n_b"] += 1
            bucket["caption"] += int(answer_hit(captions[item.file_name], item.answer))
    return [
        {
            "name": name,
            "n": counts["n"],
            "ocr_hit": counts["ocr"],
            "ocr_rate": rate(counts["ocr"], counts["n"]),
            "publish": counts["n"] >= MIN_REPORTABLE_N,
            **(
                {
                    "caption_hit": counts["caption"],
                    "caption_rate": rate(counts["caption"], counts["n_b"]),
                }
                if captions is not None
                else {}
            ),
        }
        for name, counts in sorted(buckets.items())
    ]


if __name__ == "__main__":
    raise SystemExit(main())
