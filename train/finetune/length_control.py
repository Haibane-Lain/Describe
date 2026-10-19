"""Re-score predictions under a length penalty, to separate style from quality.

The fine-tune moved mean caption length from 9.6 to 13.2 tokens. BLEU and CIDEr-D
both penalise length divergence from the references, so a length change alone
moves those numbers - in either direction - without any change in how well the
model describes an image.

This tool rescores an existing prediction file across a range of length penalties
and reports the metric curve. If a model's disadvantage disappears as the penalty
is relaxed, the gap was style. If it persists, the gap is content.

Both BLEU and CIDEr-D already apply a length penalty internally; what is varied
here is a multiplicative exponent on the brevity penalty via `nltk`-free BLEU, and
CIDEr's own length normalisation is reported alongside raw scores.

    py -3.12 -m train.finetune.length_control \
        --predictions runs/blip_base_head/predictions_full_beam5.jsonl \
        --label "E3 full decoder"
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.train.evaluate import tokenize_simple  # noqa: E402


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu_with_bp_exponent(
    hypotheses: list[list[str]],
    references: list[list[list[str]]],
    max_n: int = 4,
    bp_exponent: float = 1.0,
) -> dict:
    """Corpus BLEU where the brevity penalty is raised to ``bp_exponent``.

    ``bp_exponent = 1.0`` is standard BLEU. Lower values relax the length
    penalty, which is how a length-driven gap is told apart from a content gap.
    """
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hypothesis, refs in zip(hypotheses, references):
        hyp_len += len(hypothesis)
        # Closest reference length, the standard BLEU choice.
        ref_len += min((len(r) for r in refs), key=lambda x: (abs(x - len(hypothesis)), x))
        for n in range(1, max_n + 1):
            hyp_ngrams = ngrams(hypothesis, n)
            totals[n - 1] += max(len(hypothesis) - n + 1, 0)
            max_ref: Counter = Counter()
            for ref in refs:
                ref_ngrams = ngrams(ref, n)
                for gram, count in ref_ngrams.items():
                    max_ref[gram] = max(max_ref[gram], count)
            clipped[n - 1] += sum(min(count, max_ref[gram]) for gram, count in hyp_ngrams.items())

    precisions = []
    for n in range(max_n):
        precisions.append(clipped[n] / totals[n] if totals[n] else 0.0)

    if min(precisions) <= 0:
        bleu = 0.0
    else:
        log_sum = sum(math.log(p) for p in precisions) / max_n
        if hyp_len == 0:
            bp = 0.0
        elif hyp_len > ref_len:
            bp = 1.0
        else:
            # Standard BLEU uses exp(1 - r/c), which equals exp(-(r/c - 1)).
            # Scaling the *penalty* by the exponent keeps the direction of the
            # correction and, unlike raising a negative base to a fractional
            # power, stays real-valued: `1 - ref_len/hyp_len` is negative
            # whenever the hypothesis is shorter, and Python returns a complex
            # number for negative bases with fractional exponents.
            bp = math.exp(-bp_exponent * (ref_len / hyp_len - 1.0))
        bleu = bp * math.exp(log_sum)

    return {
        "BLEU": bleu,
        "precisions": precisions,
        "hyp_len": hyp_len,
        "ref_len": ref_len,
    }


def load(path: Path) -> tuple[list[list[str]], list[list[list[str]]]]:
    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            refs = row.get("references") or []
            if not refs:
                continue
            hypotheses.append(tokenize_simple(row.get("prediction") or ""))
            references.append([tokenize_simple(r) for r in refs])
    return hypotheses, references


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Length-controlled re-scoring.")
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--label", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    labels = args.label or [p.stem for p in args.predictions]
    if len(labels) != len(args.predictions):
        raise SystemExit("--label count must match --predictions count")

    exponents = [1.0, 0.75, 0.5, 0.25, 0.0]
    results: dict = {}

    print(f"  brevity-penalty exponent: 1.0 = standard BLEU, 0.0 = length penalty removed")
    header = "  " + f"{'model':<28}" + "".join(f"{e:>9}" for e in exponents) + f"{'hyp/ref len':>14}"
    print(header)
    for path, label in zip(args.predictions, labels):
        hypotheses, references = load(path)
        row = []
        for exponent in exponents:
            score = corpus_bleu_with_bp_exponent(hypotheses, references, bp_exponent=exponent)
            row.append(score["BLEU"])
        length_info = corpus_bleu_with_bp_exponent(hypotheses, references)
        ratio = length_info["hyp_len"] / max(length_info["ref_len"], 1)
        print(
            "  " + f"{label:<28}"
            + "".join(f"{v:>9.4f}" for v in row)
            + f"{ratio:>13.2f}x"
        )
        results[label] = {
            "bleu_by_bp_exponent": dict(zip((str(e) for e in exponents), row)),
            "hyp_len": length_info["hyp_len"],
            "ref_len": length_info["ref_len"],
            "length_ratio": ratio,
            "n_images": len(hypotheses),
        }

    print("\n  A gap that closes as the penalty relaxes is a style difference.")
    print("  A gap that survives to exponent 0.0 is a content difference.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
