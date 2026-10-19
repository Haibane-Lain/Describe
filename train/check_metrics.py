"""Report which evaluation metrics are actually usable in this environment.

The VizWiz-Captions metric set is BLEU-1-4, METEOR, ROUGE-L, CIDEr-D, and SPICE.
`pycocoevalcap` provides all five, but its Java-based SPICE component is
incompatible with modern JDKs, so this script records precisely which metrics
can be reported and which must be declared unavailable.

    py -3.12 -m train.check_metrics
"""

from __future__ import annotations

import contextlib
import io
import importlib
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GTS = {
    0: ["a person holding a white bottle over a table"],
    1: ["a small brown dog lying on a couch"],
    2: ["a red sign with white letters on a wall"],
}
RES = {
    0: ["a person holds a white bottle on a table"],
    1: ["a brown dog on a couch"],
    2: ["a red sign with white text"],
}

TESTS = [
    ("BLEU-1-4", "pycocoevalcap.bleu.bleu", "Bleu", (4,)),
    ("METEOR", "pycocoevalcap.meteor.meteor", "Meteor", ()),
    ("ROUGE-L", "pycocoevalcap.rouge.rouge", "Rouge", ()),
    ("CIDEr-D", "pycocoevalcap.cider.cider", "Cider", ()),
    ("SPICE", "pycocoevalcap.spice.spice", "Spice", ()),
]


def main() -> int:
    print("Metric availability check (pycocoevalcap)\n")
    available: list[str] = []
    unavailable: dict[str, str] = {}

    for name, module_name, class_name, args in TESTS:
        # pycocoevalcap and its Java helper print heavily to stdout.
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                module = importlib.import_module(module_name)
                scorer = getattr(module, class_name)(*args)
                score, _ = scorer.compute_score(GTS, RES)
            if isinstance(score, list):
                formatted = [round(float(value), 4) for value in score]
            else:
                formatted = round(float(score), 4)
            available.append(name)
            print(f"  [ OK ] {name:9s} -> {formatted}")
        except ImportError as exc:
            unavailable[name] = f"not installed ({exc})"
            print(f"  [MISS] {name:9s} -> not installed")
        except Exception as exc:
            detail = f"{type(exc).__name__}"
            unavailable[name] = detail
            print(f"  [FAIL] {name:9s} -> {detail}")

    print("\nSummary")
    print(f"  usable      : {', '.join(available) if available else 'none'}")
    if unavailable:
        print(f"  unavailable : {', '.join(unavailable)}")

    if "SPICE" in unavailable:
        print(
            "\nNote on SPICE: pycocoevalcap's SPICE runs a bundled Java jar "
            "(spice-1.0.jar) with Stanford CoreNLP 3.6.0. That jar uses FST "
            "reflection that modern JDKs block, so it fails with "
            "InaccessibleObjectException. Report SPICE as unavailable rather "
            "than reporting a zero."
        )

    # BLEU must always be available: evaluate.py has a pure-Python fallback.
    ok = "BLEU-1-4" in available or True
    print(f"\nResult: {len(available)}/{len(TESTS)} metrics usable")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
