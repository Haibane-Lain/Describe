"""What do blind users ask about, and does the captioner name those things?

The dataset question this answers: should we go and find or collect data for
specific object categories? That is only worth doing if there is a measurable
gap between the things users ask about and the things the model names.

Two computations, both cheap and both needed before any dataset purchase:

1. **Demand** - object terms in the 20,523 VizWiz training questions. Questions
   are what users actually wanted to know, so this is a demand-side inventory
   rather than an assumption about what blind users care about.
2. **Supply** - object terms the captioner emits, measured over its own
   predictions on the validation split.

A large demand-without-supply term is a candidate dataset target. A term that is
frequently asked about *and* frequently supplied is already handled.

    py -3.12 -m train.finetune.audit_objects \
        --predictions runs/blip_base_head/predictions_full_beam5.jsonl
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

# Interrogative and functional words: high frequency in questions, zero
# information about the object being asked about.
QUESTION_STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "would", "should", "will",
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "there",
    "here", "you", "your", "i", "my", "me", "we", "they", "them", "he", "she",
    "of", "in", "on", "at", "to", "for", "from", "with", "and", "or", "but",
    "say", "says", "said", "tell", "please", "thank", "thanks", "see", "look",
    "know", "want", "need", "help", "get", "got", "any", "some", "many", "much",
    "more", "most", "other", "another", "same", "different", "kind", "type",
    "name", "call", "called", "read", "reading", "text", "word", "words",
    "picture", "photo", "image", "thing", "things", "something", "anything",
    "if", "so", "then", "than", "as", "be", "been", "being", "have", "has",
    "had", "not", "no", "yes", "ok", "okay", "about", "into", "out", "up",
    "down", "left", "right", "front", "back", "much", "very", "just", "also",
    "one", "two", "three", "many", "all", "any", "only", "own", "new", "old",
    "good", "bad", "big", "small", "long", "short", "hi", "hello", "hey",
}

# Reading intent, expressed as question patterns rather than a hand-labelled
# category field, because the official annotations carry answer_type but not a
# content category. Multiple patterns can match one question; the counts are
# therefore indicative, not a partition.
READING_PATTERNS = [
    r"\bread\b", r"\bsay(s|ing)?\b", r"\bsays\b", r"\bwritten\b", r"\bwrite\b",
    r"\btext\b", r"\bword(s)?\b", r"\blabel\b", r"\bexpir", r"\bdate\b",
    r"\binstruction", r"\bdirection", r"\bingredient", r"\bcalorie",
    r"\bprice\b", r"\bcost\b", r"\bbrand\b", r"\btitle\b", r"\bname\b",
    r"\bnumber\b", r"\bphone\b", r"\baddress\b", r"\bmessage\b", r"\bscreen\b",
]
IDENTIFY_PATTERNS = [r"\bwhat is this\b", r"\bwhat'?s this\b", r"\bwhat is it\b", r"\bwhat are these\b"]
COLOR_PATTERNS = [r"\bcolou?r\b"]
COUNT_PATTERNS = [r"\bhow many\b", r"\bcount\b"]
FOOD_PATTERNS = [r"\bfood\b", r"\beat\b", r"\bexpir", r"\bspoil", r"\bfresh", r"\bcan\b", r"\bbottle\b", r"\bpackage\b", r"\bingredient"]
MED_PATTERNS = [r"\bmedicine\b", r"\bmedication\b", r"\bpill\b", r"\bprescription\b", r"\bdosage\b", r"\bdose\b", r"\btablet\b", r"\bpharmacy\b"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def load_questions(split: str = "train") -> list[str]:
    if not ANNOTATIONS_ZIP.exists():
        raise SystemExit(f"VQA annotations not found at {ANNOTATIONS_ZIP}")
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as archive:
        rows = json.loads(archive.read(f"{split}.json"))
    return [row.get("question") or "" for row in rows]


def load_captions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line).get("prediction") or "" for line in handle if line.strip()]


def count_terms(texts: list[str], stopwords: set[str], min_length: int = 3) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        for token in tokenize(text):
            if token in stopwords or len(token) < min_length:
                continue
            counter[token] += 1
    return counter


def match_any(questions: list[str], patterns: list[str]) -> int:
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    return sum(1 for q in questions if regex.search(q))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Object demand vs caption supply.")
    parser.add_argument("--predictions", type=Path,
                        default=Path("runs/blip_base_head/predictions_full_beam5.jsonl"))
    parser.add_argument("--questions-split", default="train", choices=["train", "val"])
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    questions = load_questions(args.questions_split)
    captions = load_captions(args.predictions)
    print(f"  questions: {len(questions):,} (VizWiz-VQA {args.questions_split})")
    print(f"  captions : {len(captions):,} ({args.predictions})")

    demand = count_terms(questions, QUESTION_STOPWORDS)
    supply = count_terms(captions, set(), min_length=3)
    n_q, n_c = max(len(questions), 1), max(len(captions), 1)

    print(f"\n  {'term':<18} {'asked':>8} {'asked%':>8} {'named':>8} {'named%':>8}  gap")
    rows = []
    for term, asked in demand.most_common(args.top):
        named = supply.get(term, 0)
        asked_pct = 100 * asked / n_q
        named_pct = 100 * named / n_c
        # Gap in percentage points: high demand, low supply is a dataset target.
        rows.append((term, asked, asked_pct, named, named_pct, asked_pct - named_pct))
        print(f"  {term:<18} {asked:>8} {asked_pct:>7.1f}% {named:>8} {named_pct:>7.1f}%  {asked_pct - named_pct:>+6.1f}")

    print("\n  intent breakdown (patterns may overlap; not a partition)")
    for label, patterns in (
        ("reading/text", READING_PATTERNS),
        ("identify", IDENTIFY_PATTERNS),
        ("colour", COLOR_PATTERNS),
        ("count", COUNT_PATTERNS),
        ("food", FOOD_PATTERNS),
        ("medicine", MED_PATTERNS),
    ):
        hits = match_any(questions, patterns)
        print(f"    {label:<14} {hits:>6}  ({100 * hits / n_q:5.1f}% of questions)")

    print("\n  top terms the model names")
    for term, count in supply.most_common(15):
        print(f"    {term:<18} {count:>8}  ({100 * count / n_c:5.1f}% of captions)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "questions_file": f"VizWiz-VQA {args.questions_split}",
                    "n_questions": len(questions),
                    "predictions_file": str(args.predictions),
                    "n_captions": len(captions),
                    "demand_vs_supply": [
                        {"term": t, "asked": a, "asked_pct": ap, "named": n, "named_pct": np_, "gap_pp": g}
                        for t, a, ap, n, np_, g in rows
                    ],
                    "intent": {
                        label: match_any(questions, patterns)
                        for label, patterns in (
                            ("reading_text", READING_PATTERNS),
                            ("identify", IDENTIFY_PATTERNS),
                            ("colour", COLOR_PATTERNS),
                            ("count", COUNT_PATTERNS),
                            ("food", FOOD_PATTERNS),
                            ("medicine", MED_PATTERNS),
                        )
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
