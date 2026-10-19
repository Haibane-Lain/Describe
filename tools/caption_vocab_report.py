"""Quantify what a captioner could realistically learn from the available captions.

Reads the VizWiz-Captions training split and reports vocabulary size, word
frequencies, and how much of the language a small model could cover. Prints a
plain-text report; nothing is written to disk.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CAP = BASE / "data/raw/vizwiz/annotations/caption/annotations"

SENTINEL = "quality issues are too severe to recognize visual content"

TEXT_WORDS = {
    "text", "word", "words", "letter", "letters", "read", "reading", "label",
    "labels", "sign", "number", "numbers", "date", "expire", "expiration",
    "price", "name", "written", "write", "print", "printed", "font", "title",
}

token_re = re.compile(r"[a-z]+")


def load(split: str):
    data = json.loads((CAP / f"{split}.json").read_text(encoding="utf-8"))
    return data["annotations"]


def main() -> None:
    for split in ("train", "val"):
        anns = load(split)
        usable = [a["caption"] for a in anns
                  if a["caption"].strip().lower() != SENTINEL]
        tokens = [t for c in usable for t in token_re.findall(c.lower())]
        types = collections.Counter(tokens)
        print(f"===== {split}: {len(anns):,} captions, "
              f"{len(usable):,} usable (sentinel rows dropped: "
              f"{len(anns) - len(usable):,})")
        print(f"  running words: {len(tokens):,}")
        print(f"  distinct words: {len(types):,}")
        for floor in (1, 2, 5, 10, 20, 50, 100, 500):
            print(f"    words seen at least {floor:>4} times: "
                  f"{sum(1 for n in types.values() if n >= floor):>6,}")
        total = len(tokens)
        for top in (50, 100, 500, 1000, 2000, 5000):
            covered = sum(n for _, n in types.most_common(top))
            print(f"    top {top:>4} words cover {100 * covered / total:5.1f}% of all words")
        print("  most common 40 words:")
        print("   ", ", ".join(f"{w}({n})" for w, n in types.most_common(40)))
        text_caps = sum(1 for c in usable if TEXT_WORDS & set(token_re.findall(c.lower())))
        print(f"  captions mentioning reading/text: {text_caps:,} "
              f"({100 * text_caps / len(usable):.1f}%)")
        lens = [len(token_re.findall(c.lower())) for c in usable]
        print(f"  caption length: mean {sum(lens) / len(lens):.1f} words, "
              f"median {sorted(lens)[len(lens) // 2]}, max {max(lens)}")
        long_caps = sum(1 for n in lens if n > 20)
        print(f"  captions longer than 20 words: {long_caps:,} "
              f"({100 * long_caps / len(usable):.1f}%)")
        print()


if __name__ == "__main__":
    main()
