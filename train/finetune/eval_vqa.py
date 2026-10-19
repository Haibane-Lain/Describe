"""Measure whether captions actually answer the questions blind users asked.

Aggregate BLEU cannot express the thing this project cares about. A caption can
score well on n-gram overlap while omitting the one detail the photographer
needed - the medicine name, the expiry date, the book title. VizWiz-VQA is the
instrument that measures it: the same photographs, taken by the same users, each
paired with a spoken question and ten crowdsourced answers.

For every answerable question we ask a blunt question of our own: **does the
caption contain the answer?** That is not a substitute for a human evaluation,
but it is a decision-relevant number, and it is comparable across models.

The unanswerable rows matter just as much in the opposite direction. 37% of
VizWiz users' questions are unanswerable from their own photograph, which is the
empirical case for the quality gate existing at all.

    py -3.12 -m train.finetune.eval_vqa \
        --predictions runs/blip_base_head/predictions_full_beam5.jsonl \
        --split val --out runs/blip_base_head/vqa_answer_coverage.json
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

# Answers that carry no visual information. "yes"/"no" say something about the
# question but nothing about the photograph, and counting them as coverage would
# inflate the figure by roughly a fifth for free.
NON_CONTENT_ANSWERS = {
    "yes", "no", "maybe", "unanswerable", "unsure", "unknown", "n/a", "na",
    "none", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
}

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "are",
    "it", "this", "that", "with", "for", "from", "by", "as", "be", "was",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", (text or "").lower())


def load_vqa(split: str, path: Path = ANNOTATIONS_ZIP) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"VQA annotations not found at {path}. Download with:\n"
            "  https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
        )
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(f"{split}.json"))


def majority_answer(row: dict) -> str | None:
    """The answer a plurality of crowdworkers gave, ignoring 'unanswerable'."""
    answers = [
        (a.get("answer") or "").strip().lower()
        for a in (row.get("answers") or [])
        if isinstance(a, dict)
    ]
    answers = [a for a in answers if a and a != "unanswerable"]
    if not answers:
        return None
    return Counter(answers).most_common(1)[0][0]


def content_terms(answer: str) -> list[str]:
    """Answer tokens worth checking for, excluding function words."""
    return [t for t in tokenize(answer) if t not in STOPWORDS]


def answer_covered(caption: str, answer: str) -> bool:
    """Does the caption contain the answer's content terms, in order?

    Substring order rather than a set: "dog years" should not be credited by a
    caption that happens to contain "years" and "dog" in unrelated places.
    """
    terms = content_terms(answer)
    if not terms:
        return False
    caption_tokens = tokenize(caption)
    if len(terms) == 1:
        return terms[0] in caption_tokens
    joined = " ".join(caption_tokens)
    return " ".join(terms) in joined


def load_predictions(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("file_name")
            if key:
                out[key] = row.get("prediction") or ""
    return out


def evaluate(predictions: dict[str, str], annotations: list[dict]) -> dict:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in annotations:
        by_image[row["image"]].append(row)

    per_type: dict[str, Counter] = defaultdict(Counter)
    quality_counts = Counter()
    unanswerable_images = 0
    covered_images = 0
    answerable_images = 0
    missed: list[dict] = []
    examples: list[dict] = []

    for image, rows in by_image.items():
        caption = predictions.get(image, "")
        answerable = [r for r in rows if int(r.get("answerable", 0)) == 1]
        if not answerable:
            unanswerable_images += 1
        else:
            answerable_images += 1
            hits = 0
            for row in answerable:
                answer = majority_answer(row)
                if answer is None:
                    continue
                hit = answer_covered(caption, answer)
                hits += int(hit)
                bucket = per_type[row.get("answer_type") or "other"]
                bucket["questions"] += 1
                bucket["covered"] += int(hit)
                if hit and len(examples) < 25:
                    examples.append(
                        {"image": image, "question": row["question"], "answer": answer, "caption": caption}
                    )
                elif not hit and len(missed) < 25:
                    missed.append(
                        {"image": image, "question": row["question"], "answer": answer, "caption": caption}
                    )
            if hits:
                covered_images += 1

        quality_counts["images"] += 1

    def rate(counter: Counter) -> float | None:
        return counter["covered"] / counter["questions"] if counter["questions"] else None

    return {
        "images_with_questions": len(by_image),
        "images_answerable": answerable_images,
        "images_unanswerable": unanswerable_images,
        "images_with_any_answer_in_caption": covered_images,
        "image_coverage_rate": covered_images / max(answerable_images, 1),
        "unanswerable_fraction": unanswerable_images / max(len(by_image), 1),
        "by_answer_type": {
            name: {
                "questions": c["questions"],
                "covered": c["covered"],
                "coverage": rate(c),
            }
            for name, c in sorted(per_type.items())
        },
        "examples_covered": examples,
        "examples_missed": missed,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Answer coverage of captions against VizWiz-VQA.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args(argv)

    predictions = load_predictions(args.predictions)
    annotations = load_vqa(args.split)
    result = evaluate(predictions, annotations)
    result["label"] = args.label or args.predictions.stem
    result["split"] = args.split
    result["predictions_file"] = str(args.predictions)

    print(f"  {result['label']}  ({args.split}, {result['images_with_questions']:,} images with questions)")
    print(f"    images answerable               {result['images_answerable']:,}")
    print(f"    images unanswerable             {result['images_unanswerable']:,} "
          f"({100 * result['unanswerable_fraction']:.1f}%)")
    print(f"    caption contains an answer      {result['images_with_any_answer_in_caption']:,} "
          f"({100 * result['image_coverage_rate']:.1f}% of answerable)")
    print("\n    by answer type")
    for name, values in result["by_answer_type"].items():
        cov = values["coverage"]
        print(f"      {name:<14} {values['covered']:>5}/{values['questions']:<5} "
              f"{'' if cov is None else f'{100 * cov:5.1f}%'}")

    if args.show:
        print(f"\n    {args.show} covered examples")
        for ex in result["examples_covered"][: args.show]:
            print(f"      Q: {ex['question'][:70]}")
            print(f"      A: {ex['answer']!r}  ->  {ex['caption'][:70]}")
        print(f"\n    {args.show} missed examples")
        for ex in result["examples_missed"][: args.show]:
            print(f"      Q: {ex['question'][:70]}")
            print(f"      A: {ex['answer']!r}  ->  {ex['caption'][:70]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n    written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
