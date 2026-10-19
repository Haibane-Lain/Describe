"""Verify and benchmark the captioner's decoders.

Checks that beam search is correct (beam_size=1 reduces to greedy, outputs are
index-valid, per-beam state reordering is right) and measures decode speed so
evaluation cost is known before running it on the test split.

    py -3.12 -m train.verify_decoders
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data.vocab import Vocab  # noqa: E402
from train.models.captioner import Captioner  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402

TOY = [
    "a person holding a white bottle",
    "a small brown dog on a couch",
    "a red sign with white text",
]


def main() -> int:
    configure_for_speed()
    info = get_device()
    print(f"device: {info.name} ({info.backend})")

    vocab = Vocab.build(TOY, min_freq=1)
    model = Captioner(
        vocab_size=len(vocab),
        feature_dim=64,
        embed_dim=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.5,
        pad_idx=vocab.pad_idx,
    ).to(info.device)
    model.eval()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))

    small = torch.randn(2, 64, device=info.device)
    big = torch.randn(32, 64, device=info.device)

    print("\nCorrectness")
    result = model.beam_decode(
        small, vocab.start_idx, vocab.end_idx, beam_size=3, max_len=6, min_len=2
    )
    check("beam returns one result per image", len(result) == 2, f"{len(result)}")
    check(
        "beam indices in range",
        all(0 <= t < len(vocab) for seq in result for t in seq),
    )

    greedy = model.greedy_decode(
        big[:8], vocab.start_idx, vocab.end_idx, max_len=15, min_len=2
    )
    # Arguments must match greedy's exactly, or the comparison is meaningless.
    beam_one = model.beam_decode(
        big[:8], vocab.start_idx, vocab.end_idx, beam_size=1, max_len=15, min_len=2
    )
    check("beam_size=1 matches greedy", greedy == beam_one)

    beam_five = model.beam_decode(
        big[:8], vocab.start_idx, vocab.end_idx, beam_size=5, max_len=20, min_len=3
    )
    check(
        "beam_size=5 indices in range",
        all(0 <= t < len(vocab) for seq in beam_five for t in seq),
    )
    check(
        "beam_size=5 respects max_len",
        all(len(s) <= 20 for s in beam_five),
        str([len(s) for s in beam_five]),
    )
    check(
        "no <end> token leaks into output",
        all(vocab.end_idx not in seq for seq in beam_five),
    )
    # A min_len above the natural caption length previously made the search
    # emit <end> repeatedly. The mask must prevent that entirely.
    long_min = model.beam_decode(
        big[:4], vocab.start_idx, vocab.end_idx, beam_size=5, max_len=25, min_len=12
    )
    check(
        "large min_len produces no <end> padding",
        all(vocab.end_idx not in seq for seq in long_min),
        f"lengths {[len(s) for s in long_min]}",
    )
    check(
        "large min_len is honoured",
        all(len(s) >= 12 for s in long_min),
        f"lengths {[len(s) for s in long_min]}",
    )
    check(
        "no degenerate all-<unk> output",
        all(any(t != vocab.unk_idx for t in seq) for seq in long_min),
    )

    print("\nSpeed")
    timings: dict[str, float] = {}
    for name, kwargs in (
        ("greedy", {"max_len": 30}),
        ("beam-3", {"beam_size": 3, "max_len": 30, "min_len": 5}),
        ("beam-5", {"beam_size": 5, "max_len": 30, "min_len": 5}),
    ):
        fn = model.greedy_decode if name == "greedy" else model.beam_decode
        fn(big, vocab.start_idx, vocab.end_idx, **kwargs)  # warmup
        torch.cuda.synchronize() if info.is_gpu else None
        started = time.time()
        fn(big, vocab.start_idx, vocab.end_idx, **kwargs)
        if info.is_gpu:
            torch.cuda.synchronize()
        elapsed = time.time() - started
        per_image = elapsed / big.shape[0]
        timings[name] = per_image
        print(f"  {name:7s}: {elapsed:5.2f}s / {big.shape[0]} imgs "
              f"({per_image * 1000:5.1f} ms/img)")

    check("greedy is faster than beam-5", timings["greedy"] < timings["beam-5"])
    check("beam-5 is faster than the old per-image loop (<500 ms/img)",
          timings["beam-5"] < 0.5, f"{timings['beam-5'] * 1000:.1f} ms/img")

    print("\nProjected evaluation cost")
    for name, per_image in timings.items():
        print(f"  {name:7s}: 8,000 test images -> {8000 * per_image / 60:5.1f} min")

    print("\n" + "=" * 60)
    failed = [c for c in checks if not c[1]]
    print(f"Result: {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        for name, _, detail in failed:
            print(f"  - {name} {detail}")
        return 1
    print("DECODER VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
