"""The reading bucket, defined in exactly one place.

``train/docs/ocr-plan.md`` hands the OCR component 577 VizWiz validation images
and 1,980 training images. Those numbers are easy to disagree with: the handoff
document's own tables report 577 images in §3 and a 575-question subtype table in
§4, because two different populations are in play and neither table says which:

* **Population A -- 577 questions over 577 images.** Every answerable validation
  question classified as reading, with a majority answer. This is what OCR must
  be measured on, because it is the population the component owns.
* **Population B -- 575 of those.** The captioner's prediction file
  (``runs/blip_large_head/predictions_full_beam5.jsonl``) covers 7,542 images and
  omits 75 validation question-images, two of which are reading questions
  (``VizWiz_val_00001985`` "What's on the screen?" -> ``black`` and
  ``VizWiz_val_00002169`` "What is visible on this screen, please?" ->
  ``nothing``). Any *comparison* against the captioner has to use the rows where
  a caption exists, or the caption is scored on an empty string.

So this module returns population A and lets the measurement tool intersect with
a prediction file to get population B. Nothing else in the project may redefine
the set -- that is how the 2,923-vs-3,048 class of disagreement started.

The classifier is imported, not reimplemented: ``is_reading_question`` and
``classify_kind`` from ``train.finetune.reading_gap`` are the ones behind the
published intent table, and ``majority_answer`` comes from
``train.finetune.eval_vqa`` for the same reason.

    py -3.12 -m train.ocr.reading_set --split val
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.eval_vqa import load_vqa, majority_answer  # noqa: E402
from train.finetune.reading_gap import classify_kind, is_reading_question  # noqa: E402

IMAGES_ROOT = Path("data/raw/vizwiz/images")

# Below this many questions a subtype rate is noise, and ocr-plan.md §4 forbids
# publishing it. The threshold lives here so every generator applies the same one.
MIN_REPORTABLE_N = 20


@dataclass(frozen=True)
class ReadingItem:
    """One answerable reading question and the image it was asked about."""

    file_name: str
    path: Path
    question: str
    answer: str
    subtype: str

    @property
    def key(self) -> str:
        return self.file_name


def reading_items(
    split: str = "val",
    *,
    images_root: Path = IMAGES_ROOT,
    strict: bool = True,
) -> list[ReadingItem]:
    """Every answerable reading question in ``split``, sorted by file name.

    ``strict`` raises when an image file is missing rather than silently
    shrinking the test set -- a measurement that quietly drops rows is worse
    than one that refuses to run.
    """
    rows = load_vqa(split)
    image_dir = images_root / split

    items: list[ReadingItem] = []
    missing: list[str] = []
    for row in rows:
        if int(row.get("answerable", 0)) != 1:
            continue
        question = row.get("question") or ""
        if not is_reading_question(question):
            continue
        answer = majority_answer(row)
        if not answer:
            continue
        file_name = row["image"]
        path = image_dir / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        items.append(
            ReadingItem(
                file_name=file_name,
                path=path,
                question=question,
                answer=answer,
                subtype=classify_kind(question),
            )
        )

    if missing and strict:
        preview = ", ".join(sorted(missing)[:5])
        raise SystemExit(
            f"{len(missing)} reading image(s) missing under {image_dir} "
            f"(e.g. {preview}). Refusing to measure on an incomplete set."
        )

    items.sort(key=lambda item: item.file_name)
    return items


def reading_set_stats(items: list[ReadingItem]) -> dict:
    """Counts and the per-subtype breakdown, with the publication flag applied."""
    by_subtype = Counter(item.subtype for item in items)
    return {
        "questions": len(items),
        "images": len({item.file_name for item in items}),
        "subtypes": [
            {
                "name": name,
                "n": n,
                "publish": n >= MIN_REPORTABLE_N,
            }
            for name, n in by_subtype.most_common()
        ],
        "suppressed_subtypes": [
            name for name, n in by_subtype.most_common() if n < MIN_REPORTABLE_N
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Describe the OCR reading test set.")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--no-strict", action="store_true",
                        help="allow missing image files instead of failing")
    args = parser.parse_args(argv)

    items = reading_items(args.split, strict=not args.no_strict)
    stats = reading_set_stats(items)

    print(f"  split            : {args.split}")
    print(f"  questions        : {stats['questions']:,}")
    print(f"  distinct images  : {stats['images']:,}")
    print()
    print(f"  {'subtype':<26} {'n':>5}  publish")
    for row in stats["subtypes"]:
        flag = "yes" if row["publish"] else "NO -- n < 20"
        print(f"  {row['name']:<26} {row['n']:>5}  {flag}")
    print()
    print(f"  total            : {sum(r['n'] for r in stats['subtypes']):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
