"""The intent table: one consistent measurement of every question category.

Two different intent breakdowns existed in this project and they did not agree —
one totalled 2,923 questions and another 3,048, because they classified and scored
differently. Publishing either without the other is how a report ends up with
numbers that don't add up.

This computes all intents in **one pass with one scoring rule**, using the same
matcher as `reading_gap` and `eval_vqa`, so the rows sum to the total.

    py -3.12 -m train.finetune.intent_table
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.eval_vqa import answer_covered, load_vqa, majority_answer  # noqa: E402
from train.finetune.reading_gap import classify_kind, is_reading_question  # noqa: E402

COLOR = re.compile(r"colou?r", re.IGNORECASE)
COUNT = re.compile(r"how many|count", re.IGNORECASE)
IDENTIFY = re.compile(r"what is this|what'?s this|what is it|what are these", re.IGNORECASE)

# Order matters: reading is tested first, so a question mentioning "number" is
# counted as reading rather than falling into a generic bucket.
ORDER = ["colour", "identify object", "other", "reading / detail", "count"]


def classify(question: str) -> str:
    if is_reading_question(question):
        return "reading / detail"
    if COLOR.search(question):
        return "colour"
    if COUNT.search(question):
        return "count"
    if IDENTIFY.search(question):
        return "identify object"
    return "other"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Consistent intent table over VizWiz-VQA val.")
    parser.add_argument("--predictions", type=Path,
                        default=Path("runs/blip_large_head/predictions_full_beam5.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    predictions: dict[str, str] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["file_name"]] = row.get("prediction") or ""

    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in load_vqa("val"):
        by_image[row["image"]].append(row)

    tally: dict[str, Counter] = defaultdict(Counter)
    subtypes: dict[str, Counter] = defaultdict(Counter)

    for image, rows in by_image.items():
        caption = predictions.get(image)
        if caption is None:
            continue
        for row in rows:
            if int(row.get("answerable", 0)) != 1:
                continue
            answer = majority_answer(row)
            if not answer:
                continue
            kind = classify(row["question"])
            hit = int(answer_covered(caption, answer))
            tally[kind]["n"] += 1
            tally[kind]["hit"] += hit
            if kind == "reading / detail":
                sub = classify_kind(row["question"])
                subtypes[sub]["n"] += 1
                subtypes[sub]["hit"] += hit

    total_n = sum(t["n"] for t in tally.values())
    total_hit = sum(t["hit"] for t in tally.values())

    print(f"  questions scored: {total_n:,}  (answerable, with a majority answer)")
    print()
    print(f"  {'intent':<18} {'n':>6} {'hit':>6} {'rate':>8}")
    for kind in ORDER:
        t = tally[kind]
        print(f"  {kind:<18} {t['n']:>6} {t['hit']:>6} "
              f"{100 * t['hit'] / max(t['n'], 1):>7.1f}%")
    print(f"  {'TOTAL':<18} {total_n:>6} {total_hit:>6} "
          f"{100 * total_hit / max(total_n, 1):>7.1f}%")

    print()
    print(f"  {'reading subtype':<24} {'n':>5} {'hit':>5} {'rate':>8}")
    for sub, t in sorted(subtypes.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {sub:<24} {t['n']:>5} {t['hit']:>5} "
              f"{100 * t['hit'] / max(t['n'], 1):>7.1f}%")

    reading = tally["reading / detail"]
    non_reading = total_n - reading["n"]
    print()
    print(f"  reading      {reading['hit']:>4}/{reading['n']:<5} "
          f"{100 * reading['hit'] / max(reading['n'], 1):>5.1f}%")
    print(f"  non-reading  {total_hit - reading['hit']:>4}/{non_reading:<5} "
          f"{100 * (total_hit - reading['hit']) / max(non_reading, 1):>5.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "predictions": str(args.predictions),
                    "questions_scored": total_n,
                    "intents": {
                        k: {"n": tally[k]["n"], "hit": tally[k]["hit"]} for k in ORDER
                    },
                    "reading_subtypes": {
                        k: {"n": v["n"], "hit": v["hit"]} for k, v in subtypes.items()
                    },
                    "note": "single pass, single scoring rule; rows sum to the total",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
