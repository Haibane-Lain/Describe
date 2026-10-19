"""Build a caption-training slice from VizWiz-VQA's informative answers.

VizWiz-VQA as a whole is unusable for caption training: 86.3% of its answers are
one or two words, the most common being `no`, `yes` and bare colour words, while
real VizWiz captions average 11.5 words. Training on all of it would teach answer
fragments and destroy fluency.

But 8,742 of the 14,991 answerable training pairs answer with something that
*names the object* - "basil leaves", "coca cola", "solar garden light", "monitor
cleaning kit", "dr pepper", "80 cents". That slice is the only available source
of detail-on-an-object supervision on the same photographs, from the same
photographers, as the deployment domain.

This tool writes that slice as a manifest so it can be mixed into training, and
it reports what it excluded. It does **not** train anything: the risk that
answer-derived captions teach a degenerate "a photo of {noun}" template has to be
tested by measuring answer coverage *and* fluency, not assumed away.

    py -3.12 -m train.finetune.build_vqa_slice --out data/manifests/vqa_informative_train.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ANNOTATIONS_ZIP = Path("data/raw/vizwiz/archives/VQA_Annotations.zip")

# Answers that carry no information about the photograph. Kept as a set rather
# than a length rule so the exclusion is explicit and reviewable.
NON_INFORMATIVE = {
    "yes", "no", "maybe", "unanswerable", "none", "unsure", "unknown", "n/a",
    "na", "not sure", "cant tell", "can't tell", "blurry", "too dark",
    "insufficient photo quality", "cannot tell", "unclear", "nothing",
}

COLORS = {
    "white", "black", "grey", "gray", "red", "blue", "green", "yellow",
    "orange", "purple", "pink", "brown", "silver", "gold", "tan", "beige",
    "maroon", "navy", "teal", "violet", "clear", "transparent",
}

# Captions are emitted as a sentence so the model sees grammatical text rather
# than a bare noun phrase. "A photo of X" is the convention the BLIP family is
# pretrained with, so this matches the model's existing output distribution.
TEMPLATE = "a photo of {answer}"


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


def is_informative(answer: str, min_words: int = 2) -> tuple[bool, str]:
    """Is this answer worth training on, and if not, why not."""
    if answer in NON_INFORMATIVE:
        return False, "non-informative phrase"
    if answer in COLORS:
        return False, "bare colour"
    if re.fullmatch(r"[\d\s.,/:-]+", answer):
        return False, "numeric only"
    words = re.findall(r"[a-z0-9]+", answer)
    if len(words) < min_words:
        return False, f"fewer than {min_words} words"
    if len(answer) > 80:
        return False, "over-long answer"
    return True, ""


def load_annotations(split: str, path: Path = ANNOTATIONS_ZIP) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"VQA annotations not found at {path}. Download with:\n"
            "  https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip"
        )
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(f"{split}.json"))


def build(split: str, *, min_words: int, require_answerable: bool) -> tuple[list[dict], Counter]:
    rows = load_annotations(split)
    kept: list[dict] = []
    reasons: Counter = Counter()

    for row in rows:
        if require_answerable and int(row.get("answerable", 0)) != 1:
            reasons["not answerable"] += 1
            continue
        answer = majority_answer(row)
        if answer is None:
            reasons["no majority answer"] += 1
            continue
        ok, why = is_informative(answer, min_words)
        if not ok:
            reasons[why] += 1
            continue
        kept.append(
            {
                "file_name": row["image"],
                "image_id": Path(row["image"]).stem,
                "captions": [TEMPLATE.format(answer=answer)],
                "question": row.get("question"),
                "answer": answer,
                "source": f"vizwiz-vqa-{split}-informative",
                "synthetic_caption": True,
            }
        )
    return kept, reasons


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the informative VizWiz-VQA caption slice.")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--include-unanswerable", action="store_true",
                        help="also convert unanswerable rows (their answers are refusal phrases)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    kept, reasons = build(
        args.split, min_words=args.min_words, require_answerable=not args.include_unanswerable
    )
    total = len(load_annotations(args.split))

    print(f"  VizWiz-VQA {args.split}: {total:,} annotations")
    print(f"  kept as caption supervision: {len(kept):,} ({100 * len(kept) / max(total, 1):.1f}%)")
    print("\n  excluded")
    for reason, count in reasons.most_common():
        print(f"    {reason:<24} {count:>7,}")

    if kept:
        print("\n  samples")
        for row in kept[:10]:
            print(f"    {row['question'][:44]:<46} -> {row['captions'][0]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    print(f"\n  written: {args.out} ({len(kept):,} rows)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "annotations": total,
                    "kept": len(kept),
                    "excluded": dict(reasons),
                    "template": TEMPLATE,
                    "min_words": args.min_words,
                    "note": "captions are synthetic, derived from crowdworker answers, not human descriptions",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
