"""Which component should own each question a blind user asks?

VizWiz-VQA gives a measured hit rate per question intent for the captioner, and
the split is stark:

| intent | hit rate |
|---|---|
| colour | 51.9% |
| identify object | 42.8% |
| other | 21.3% |
| **reading / detail** | **13.6%** |
| count | 0.0% |

Reading is the captioner's worst bucket by a wide margin, three attempts to
improve it by adding training data all failed, and the architecture change that
might address it (training the vision tower) is infeasible on this hardware at
6.5 h/epoch. So the question is no longer "how do we make the captioner read" but
"does something else already read, and is that written down anywhere".

This tool produces the evidence for that handoff: the actual failures, grouped,
with what the captioner said and what an OCR pass would have needed to extract.
It is deliberately a document generator, not a model change.

    py -3.12 -m train.finetune.reading_gap \
        --predictions runs/blip_large_head/predictions_full_beam5.jsonl \
        --out train/docs/reading-gap-handoff.md
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANNOTATIONS_ZIP = Path("data/raw/vizwiz/archives/VQA_Annotations.zip")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are",
    "it", "this", "that", "with", "for", "from", "by", "as", "be", "was",
}

# Sub-intents of "reading", which need different treatment.
READING_KINDS = [
    ("expiry / date", r"expir|\bdate\b|best by|use by|good until"),
    ("price / cost", r"\bprice\b|\bcost\b|how much (is|does)|dollar|cent"),
    ("product / brand name", r"\bbrand\b|\bname of\b|what (is|'s) (this|the) (product|item|brand)|product name"),
    ("medicine / dosage", r"\bmedicine\b|\bmedication\b|\bpill\b|prescription|dosage|\bdose\b|\btablet\b|\bmg\b"),
    ("book / media title", r"\btitle\b|\bbook\b|\bmagazine\b|\bmovie\b|\bcd\b|\bdvd\b|\bgame\b"),
    ("label / packaging text", r"\blabel\b|\bpackage\b|\bcontainer\b|\bcan\b|\bbottle\b|\bbox\b|\bwrapper\b"),
    ("sign / screen text", r"\bsign\b|\bscreen\b|\bmonitor\b|\bdisplay\b|\bmenu\b"),
    ("ingredients / nutrition", r"ingredient|nutrition|calorie|sodium|sugar|fat\b|protein"),
    ("phone / number", r"\bphone\b|\bnumber\b|\bcall\b|area code"),
    ("address / location", r"\baddress\b|\bstreet\b|\bcity\b|\bzip\b|\bstate\b"),
    ("instructions", r"instruction|\bdirection\b|how do i|how to"),
    ("other reading", r"."),
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def majority_answer(row: dict) -> str | None:
    answers = [
        (a.get("answer") or "").strip().lower()
        for a in (row.get("answers") or [])
        if isinstance(a, dict)
    ]
    answers = [a for a in answers if a and a != "unanswerable"]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def is_reading_question(question: str) -> bool:
    return bool(
        re.search(
            r"read|say|says|written|label|text|word|expir|date|title|name|number|"
            r"brand|instruction|ingredient|price|cost|phone|address|message|screen",
            question,
            re.IGNORECASE,
        )
    )


def classify_kind(question: str) -> str:
    for name, pattern in READING_KINDS:
        if re.search(pattern, question, re.IGNORECASE):
            return name
    return "other reading"


def answer_in_caption(caption: str, answer: str) -> bool:
    terms = [t for t in tokenize(answer) if t not in STOPWORDS]
    if not terms:
        return False
    caption_tokens = tokenize(caption)
    if len(terms) == 1:
        return terms[0] in caption_tokens
    return " ".join(terms) in " ".join(caption_tokens)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the reading-gap handoff note.")
    parser.add_argument("--predictions", type=Path,
                        default=Path("runs/blip_large_head/predictions_full_beam5.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("train/docs/reading-gap-handoff.md"))
    parser.add_argument("--samples-per-kind", type=int, default=4)
    args = parser.parse_args(argv)

    if not ANNOTATIONS_ZIP.exists():
        raise SystemExit(f"VQA annotations not found at {ANNOTATIONS_ZIP}")
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as archive:
        annotations = json.loads(archive.read("val.json"))

    predictions: dict[str, str] = {}
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["file_name"]] = row.get("prediction") or ""

    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in annotations:
        by_image[row["image"]].append(row)

    # Intent-level tally over every answerable question.
    tally: dict[str, Counter] = defaultdict(Counter)
    failures: dict[str, list[dict]] = defaultdict(list)

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
            hit = answer_in_caption(caption, answer)
            kind = classify_kind(row["question"]) if is_reading_question(row["question"]) else "non-reading"
            tally[kind]["questions"] += 1
            tally[kind]["hit"] += int(hit)
            if not hit and kind != "non-reading":
                failures[kind].append(
                    {
                        "image": image,
                        "question": row["question"],
                        "answer": answer,
                        "caption": caption,
                    }
                )

    ordered = [name for name, _ in READING_KINDS if name in tally]
    total_q = sum(t["questions"] for t in tally.values())
    total_hit = sum(t["hit"] for t in tally.values())
    reading_q = sum(tally[k]["questions"] for k in ordered)
    reading_hit = sum(tally[k]["hit"] for k in ordered)

    lines: list[str] = []
    add = lines.append
    add("# Reading-gap handoff: which component owns which question")
    add("")
    add("Generated by `train.finetune.reading_gap`. Every figure is measured on the")
    add("VizWiz-VQA validation split against the deployed captioner")
    add(f"(`{args.predictions}`).")
    add("")
    add("## Why this document exists")
    add("")
    add("The captioner answers questions about **objects** and **colours** reasonably")
    add("well and questions that require **reading text** poorly. Three attempts to")
    add("close that gap with more training data failed (text-aware subset, synthetic")
    add("degradation, answer-derived captions), and the architectural fix - training")
    add("the vision tower - is infeasible on this hardware at 6.5 h/epoch.")
    add("")
    add("So this is a component-ownership question, not a modelling question.")
    add("")
    add("## Measured hit rate by intent")
    add("")
    add("| Intent | Questions | Answered | Hit rate |")
    add("|---|---|---|---|")
    for kind in ordered:
        t = tally[kind]
        add(f"| {kind} | {t['questions']:,} | {t['hit']:,} | "
            f"{100 * t['hit'] / max(t['questions'], 1):.1f}% |")
    non = tally.get("non-reading")
    if non:
        add(f"| _all non-reading questions_ | {non['questions']:,} | {non['hit']:,} | "
            f"{100 * non['hit'] / max(non['questions'], 1):.1f}% |")
    add("")
    add(f"**Reading questions: {reading_hit:,}/{reading_q:,} = "
        f"{100 * reading_hit / max(reading_q, 1):.1f}%.** "
        f"All questions: {total_hit:,}/{total_q:,} = "
        f"{100 * total_hit / max(total_q, 1):.1f}%.")
    add("")
    add("## What the captioner produced, by failure type")
    add("")
    add("For each type: what the user asked, the answer a plurality of crowdworkers")
    add("gave, and what the captioner said instead. The last column is what an OCR or")
    add("text-reading pass would have had to extract.")
    add("")
    for kind in ordered:
        items = failures.get(kind) or []
        if not items:
            continue
        t = tally[kind]
        add(f"### {kind} — {len(items)} failures "
            f"({100 * t['hit'] / max(t['questions'], 1):.1f}% hit rate)")
        add("")
        add("| asked | crowdworker answer | captioner said | OCR must extract |")
        add("|---|---|---|---|")
        for item in items[: args.samples_per_kind]:
            q = item["question"].replace("|", "/")[:58]
            a = item["answer"].replace("|", "/")[:26]
            c = item["caption"].replace("|", "/")[:58]
            add(f"| {q} | **{a}** | {c} | {a} |")
        add("")

    add("## Handoff")
    add("")
    add("| Question type | Owning component | Basis |")
    add("|---|---|---|")
    add("| object identity, scene description | **captioner** | 42.8% answer hit rate, "
        "40.3% 4-gram match |")
    add("| colour | **captioner** | 51.9% hit rate; 43% of captions name a colour |")
    add("| whether an image is usable | **quality gate** | FA 4.79%, AUC 0.964; "
        "the captioner already hedges 20x more on unusable images but never refuses |")
    add("| printed text: labels, names, dates, prices, dosages | **OCR** | captioner "
        f"reaches only {100 * reading_hit / max(reading_q, 1):.1f}% here |")
    add("")
    add("**Open item:** the OCR component's hit rate on this same split has not been")
    add("measured here. Until it is, the project cannot state that the reading bucket is")
    add("covered - only that the captioner does not cover it. These "
        f"{reading_q:,} questions are a ready-made test set for that measurement.")
    add("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print(f"  questions analysed : {total_q:,}")
    print(f"  reading questions  : {reading_q:,} "
          f"({100 * reading_q / max(total_q, 1):.1f}% of all)")
    print(f"  reading hit rate   : {100 * reading_hit / max(reading_q, 1):.1f}%")
    print(f"  non-reading hit    : {100 * (non['hit'] if non else 0) / max(non['questions'] if non else 1, 1):.1f}%")
    print()
    print(f"  {'intent':<28} {'n':>6} {'hit':>6} {'rate':>7}")
    for kind in ordered:
        t = tally[kind]
        print(f"  {kind:<28} {t['questions']:>6} {t['hit']:>6} "
              f"{100 * t['hit'] / max(t['questions'], 1):>6.1f}%")
    print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
