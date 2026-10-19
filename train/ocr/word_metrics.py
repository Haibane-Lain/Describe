"""Word-level correctness on text images (proposal §5C).

The VizWiz reading questions measure whether OCR *answers a blind user's
question*, which is the project's primary instrument but a blunt one: the answer
is a crowdworker majority, not verified text, and the scoring rule only asks
whether the answer string is present. Proposal §5C asks for something different
and narrower -- word-level correctness on photographs that actually contain
text -- and `train/docs/build-plan.md` §2.4 notes that this set **has to be
collected by the team**, because no existing dataset provides label and document
photographs taken for the purpose.

This module is the scoring half of that, ready before the photographs exist:

    data/raw/ocr_eval/ground_truth.jsonl   {"file_name": ..., "ground_truth": ...}
    data/raw/ocr_eval/images/<file_name>

    py -3.12 -m train.ocr.word_metrics
    py -3.12 -m train.ocr.word_metrics --predictions runs/ocr_easyocr/regions_x.jsonl

Character error rate (CER) and word error rate (WER) are reported both
micro-averaged (total edits / total reference length, which is the honest
corpus-level figure) and per image. Exact match is reported after
normalisation, because "MLK 250 ML" against "MLK 250ml" is not a reading error.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

EVAL_ROOT = Path("data/raw/ocr_eval")
GROUND_TRUTH = EVAL_ROOT / "ground_truth.jsonl"
IMAGES = EVAL_ROOT / "images"

NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Casefold and reduce to single-spaced alphanumerics."""
    return " ".join(NORMALIZE_PATTERN.sub(" ", (text or "").lower()).split())


def _edit_distance(reference: list, hypothesis: list) -> int:
    """Levenshtein distance over two sequences (characters or words)."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,               # deletion
                    current[j - 1] + 1,            # insertion
                    previous[j - 1] + (ref_item != hyp_item),  # substitution
                )
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on normalised text; 0.0 when both are empty."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalised text; 0.0 when both are empty."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def exact_match(reference: str, hypothesis: str) -> bool:
    """Whether the normalised strings agree exactly."""
    return normalize(reference) == normalize(hypothesis)


@dataclass(frozen=True)
class Pair:
    file_name: str
    ground_truth: str


def load_pairs(path: Path = GROUND_TRUTH) -> list[Pair]:
    if not path.exists():
        raise SystemExit(
            f"No ground truth at {path}.\n"
            "  The label and document photographs for proposal section 5C are a team\n"
            "  collection task (build-plan section 2.4: 'the team must photograph labels\n"
            "  and documents'). This tool scores them; it cannot invent them.\n"
            f"  Expected: {path} with one {{\"file_name\", \"ground_truth\"}} per line,\n"
            f"  and the images under {IMAGES}/."
        )
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs.append(
                Pair(file_name=row["file_name"], ground_truth=row.get("ground_truth") or "")
            )
    return pairs


def evaluate(pairs: list[Pair], hypotheses: dict[str, str]) -> dict:
    """Micro and per-image CER / WER / exact match."""
    rows: list[dict] = []
    total_char_edits = 0
    total_char_ref = 0
    total_word_edits = 0
    total_word_ref = 0

    for pair in pairs:
        hypothesis = hypotheses.get(pair.file_name, "")
        ref_chars = list(normalize(pair.ground_truth))
        hyp_chars = list(normalize(hypothesis))
        ref_words = normalize(pair.ground_truth).split()
        hyp_words = normalize(hypothesis).split()
        char_edits = _edit_distance(ref_chars, hyp_chars)
        word_edits = _edit_distance(ref_words, hyp_words)
        total_char_edits += char_edits
        total_char_ref += len(ref_chars)
        total_word_edits += word_edits
        total_word_ref += len(ref_words)
        rows.append(
            {
                "file_name": pair.file_name,
                "ground_truth": pair.ground_truth,
                "hypothesis": hypothesis,
                "cer": round(char_edits / len(ref_chars), 4) if ref_chars else None,
                "wer": round(word_edits / len(ref_words), 4) if ref_words else None,
                "exact_match": exact_match(pair.ground_truth, hypothesis),
            }
        )

    return {
        "images": len(pairs),
        "cer_micro": total_char_edits / total_char_ref if total_char_ref else None,
        "wer_micro": total_word_edits / total_word_ref if total_word_ref else None,
        "exact_match_rate": (
            sum(1 for row in rows if row["exact_match"]) / len(rows) if rows else None
        ),
        "per_image": rows,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Word-level OCR accuracy on a collected set.")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="score saved predictions instead of running OCR now")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    pairs = load_pairs(args.ground_truth)

    hypotheses: dict[str, str] = {}
    if args.predictions is not None:
        with args.predictions.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                hypotheses[row["file_name"]] = row.get("text") or row.get("prediction") or ""
    else:
        from train.ocr.engine import EasyOcrBackend, OcrEngine

        engine = OcrEngine(EasyOcrBackend(), pipeline="word_metrics")
        for index, pair in enumerate(pairs, start=1):
            path = args.images / pair.file_name
            if not path.exists():
                raise SystemExit(f"image missing: {path}")
            hypotheses[pair.file_name] = engine.read_text(path).text
            if index % 10 == 0 or index == len(pairs):
                print(f"    {index}/{len(pairs)}", flush=True)

    result = evaluate(pairs, hypotheses)
    cer_micro = result["cer_micro"]
    wer_micro = result["wer_micro"]
    match_rate = result["exact_match_rate"]

    print()
    print(f"  images        : {result['images']}")
    print(f"  CER (micro)   : {'n/a' if cer_micro is None else format(100 * cer_micro, '.1f') + '%'}")
    print(f"  WER (micro)   : {'n/a' if wer_micro is None else format(100 * wer_micro, '.1f') + '%'}")
    print(f"  exact match   : {'n/a' if match_rate is None else format(100 * match_rate, '.1f') + '%'}")
    print()
    for row in result["per_image"][:20]:
        cer_text = "  n/a" if row["cer"] is None else format(100 * row["cer"], "5.1f") + "%"
        print(f"    {row['file_name'][:34]:<34} CER {cer_text}  exact={row['exact_match']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
