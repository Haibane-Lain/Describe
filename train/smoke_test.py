"""Smoke test: prove the environment and the training loop work before any data.

Run this before downloading datasets. It uses synthetic features and a toy
vocabulary, so a failure here means an environment or code problem, never a
data problem.

    py -3.12 -m train.smoke_test
    py -3.12 -m train.smoke_test --device cpu     # force the CPU path
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import torch

# Allow `python train/smoke_test.py` as well as `-m train.smoke_test`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data.vocab import Vocab  # noqa: E402
from train.models.captioner import Captioner, collate_captions  # noqa: E402
from train.utils.device import (  # noqa: E402
    check_rocm_compatibility,
    configure_for_speed,
    get_device,
    set_seed,
)

TOY_CAPTIONS = [
    "a person holding a white bottle",
    "a small brown dog on a couch",
    "a red sign with white text",
    "an open laptop on a wooden table",
    "a cup of coffee next to a book",
    "a blue chair in a bright room",
    "a green plant by the window",
    "a black cat sleeping on a bed",
    "a yellow taxi on a busy street",
    "a plate of food on a table",
]

FEATURE_DIM = 64
NUM_SYNTHETIC_IMAGES = 64


class Checks:
    """Collects pass/fail results so every check runs even if one fails."""

    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.results.append((name, bool(condition), detail))
        status = "PASS" if condition else "FAIL"
        line = f"  [{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line, flush=True)
        return bool(condition)

    @property
    def failed(self) -> list[tuple[str, bool, str]]:
        return [r for r in self.results if not r[1]]

    def summary(self) -> str:
        total = len(self.results)
        bad = len(self.failed)
        return f"{total - bad}/{total} checks passed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Captioner smoke test")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="force a compute device; 'auto' prefers the GPU",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    checks = Checks()
    set_seed(args.seed)

    # ------------------------------------------------------------- 1. device
    print("\n1. Device resolution", flush=True)
    prefer_gpu = args.device != "cpu"
    info = get_device(prefer_gpu=prefer_gpu)
    print(info.describe(), flush=True)
    print(flush=True)

    if args.device == "gpu":
        checks.check(
            "GPU requested and available",
            info.is_gpu,
            f"resolved to {info.device}",
        )
    else:
        checks.check("device resolved", True, f"{info.device} ({info.backend})")

    for warning in check_rocm_compatibility(info):
        print(f"  [WARN] {warning}", flush=True)

    applied = configure_for_speed()
    for key, value in applied.items():
        print(f"  [SET ] {key} = {value}", flush=True)
    device = info.device

    # ----------------------------------------------------------- 2. vocabulary
    print("\n2. Vocabulary", flush=True)
    vocab = Vocab.build(TOY_CAPTIONS, min_freq=1, max_size=1000)
    checks.check("vocab built", len(vocab) > len(TOY_CAPTIONS) // 2, f"{len(vocab)} tokens")
    checks.check("special tokens present", all(
        tok in vocab.word2idx for tok in ("<pad>", "<start>", "<end>", "<unk>")
    ))

    encoded = vocab.encode(TOY_CAPTIONS[0])
    checks.check("encode starts with <start>", encoded[0] == vocab.start_idx)
    checks.check("encode ends with <end>", encoded[-1] == vocab.end_idx)

    round_trip = vocab.decode(encoded)
    checks.check(
        "decode round-trips",
        round_trip.lower().rstrip(".") == TOY_CAPTIONS[0].lower(),
        repr(round_trip),
    )

    # --------------------------------------------------------------- 3. model
    print("\n3. Model construction and forward pass", flush=True)
    model = Captioner(
        vocab_size=len(vocab),
        feature_dim=FEATURE_DIM,
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        pad_idx=vocab.pad_idx,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    checks.check("model built", param_count > 0, f"{param_count:,} parameters")

    captions = collate_captions(vocab.encode_batch(TOY_CAPTIONS[:8]), vocab.pad_idx)
    features = torch.randn(8, FEATURE_DIM, device=device)
    logits = model(features, captions[:, :-1].to(device))
    checks.check(
        "forward pass shape",
        tuple(logits.shape) == (8, captions.shape[1] - 1, len(vocab)),
        f"{tuple(logits.shape)}",
    )
    checks.check(
        "on requested device",
        str(logits.device).startswith(str(device.type)),
        str(logits.device),
    )

    # ------------------------------------------------------------ 4. training
    print("\n4. Training loop (synthetic, 2 epochs)", flush=True)
    torch.manual_seed(args.seed)
    num_images = NUM_SYNTHETIC_IMAGES
    train_features = torch.randn(num_images, FEATURE_DIM, device=device)
    train_captions = collate_captions(
        vocab.encode_batch(
            [TOY_CAPTIONS[i % len(TOY_CAPTIONS)] for i in range(num_images)]
        ),
        vocab.pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_history: list[float] = []
    batch_size = 16

    model.train()
    for epoch in range(2):
        permutation = torch.randperm(num_images, device=device)
        epoch_losses: list[float] = []
        for start in range(0, num_images, batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(train_features[idx], train_captions[idx])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            loss_history.append(float(loss.item()))

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"  epoch {epoch + 1}: mean loss {mean_loss:.4f}", flush=True)

    checks.check("loss is finite", all(
        v == v and abs(v) != float("inf") for v in loss_history
    ))

    quarter = max(1, len(loss_history) // 4)
    first = sum(loss_history[:quarter]) / quarter
    last = sum(loss_history[-quarter:]) / quarter
    checks.check(
        "loss decreases",
        last < first,
        f"{first:.4f} -> {last:.4f}",
    )

    embed_grad = model.embed.weight.grad
    checks.check(
        "embedding received gradient",
        embed_grad is not None and float(embed_grad.abs().sum().item()) > 0,
    )

    # -------------------------------------------------------------- 5. decode
    print("\n5. Decoding", flush=True)
    sample_features = train_features[:3]
    greedy = model.greedy_decode(
        sample_features, vocab.start_idx, vocab.end_idx, max_len=15
    )
    checks.check("greedy returns one result per image", len(greedy) == 3)
    checks.check(
        "greedy output is index-valid",
        all(
            all(0 <= t < len(vocab) for t in seq)
            for seq in greedy
        ),
    )
    for i, seq in enumerate(greedy):
        print(f"  greedy[{i}]: {vocab.decode(seq)!r}", flush=True)

    beam = model.beam_decode(
        sample_features[:1], vocab.start_idx, vocab.end_idx,
        beam_size=3, max_len=15, min_len=2,
    )
    checks.check("beam returns one result per image", len(beam) == 1)
    checks.check(
        "beam output is index-valid",
        all(0 <= t < len(vocab) for t in beam[0]),
    )
    print(f"  beam[0]  : {vocab.decode(beam[0])!r}", flush=True)

    # ---------------------------------------------------------- 6. checkpoint
    print("\n6. Checkpoint round-trip", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "smoke.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": len(vocab),
                "config": {
                    "feature_dim": FEATURE_DIM,
                    "embed_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "dropout": 0.3,
                },
            },
            ckpt_path,
        )
        checks.check("checkpoint written", ckpt_path.exists(), f"{ckpt_path.stat().st_size:,} bytes")

        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        restored = Captioner(
            vocab_size=payload["vocab_size"], pad_idx=vocab.pad_idx, **payload["config"]
        ).to(device)
        restored.load_state_dict(payload["model_state"])
        restored.eval()

        model.eval()
        with torch.no_grad():
            before = model.greedy_decode(
                sample_features[:2], vocab.start_idx, vocab.end_idx, max_len=10
            )
            after = restored.greedy_decode(
                sample_features[:2], vocab.start_idx, vocab.end_idx, max_len=10
            )
        checks.check("restored model reproduces captions", before == after)

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 62, flush=True)
    print(f"Device used: {info.name} ({info.backend})", flush=True)
    print(f"Result: {checks.summary()}", flush=True)
    if checks.failed:
        print("\nFailed checks:", flush=True)
        for name, _, detail in checks.failed:
            print(f"  - {name} {detail}", flush=True)
        print("=" * 62, flush=True)
        return 1
    print("SMOKE TEST PASSED", flush=True)
    print("=" * 62, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\nSMOKE TEST CRASHED", flush=True)
        traceback.print_exc()
        raise SystemExit(2)
