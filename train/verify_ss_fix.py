"""Verify that scheduled sampling no longer collapses the attention model.

Trains briefly at a deliberately high sampling ratio (where the old code always
collapsed) and reports whether the loss keeps moving and prediction diversity
survives.

    py -3.12 -m train.verify_ss_fix
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data.dataset import load_feature_cache  # noqa: E402
from train.data.vocab import Vocab  # noqa: E402
from train.models.attention_captioner import AttentionCaptioner  # noqa: E402
from train.models.captioner import collate_captions  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


def main() -> int:
    configure_for_speed()
    info = get_device()
    torch.manual_seed(1337)
    print(f"device: {info.name}\n")

    manifest = Path("data/manifests/flickr30k.jsonl")
    index = Path("data/features/flickr30k_resnet101_spatial.jsonl")
    features, rows = load_feature_cache(index, mmap=True)
    rows = [r for r in rows if r.get("captions")][:512]

    vocab_rows = [json.loads(line) for line in manifest.open(encoding="utf-8")
                  if line.strip()][:20000]
    captions = [c for r in vocab_rows for c in (r.get("captions") or [])]
    vocab = Vocab.build(captions, min_freq=5, max_size=10000)
    print(f"vocab {len(vocab):,} | images {len(rows)}")

    indices = [r["feature_row"] for r in rows]
    feats = torch.from_numpy(
        np.asarray(features[indices], dtype=np.float32)
    ).to(info.device)
    caps = collate_captions(
        [vocab.encode(r["captions"][0]) for r in rows], vocab.pad_idx
    ).to(info.device)
    print(f"batch {tuple(feats.shape)} captions {tuple(caps.shape)}\n")

    model = AttentionCaptioner(
        vocab_size=len(vocab), feature_dim=2048, embed_dim=256, hidden_dim=256,
        attn_dim=256, dropout=0.3, pad_idx=vocab.pad_idx,
    ).to(info.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=info.device).manual_seed(1337)

    # A high ratio is where the old code collapsed within a few hundred steps.
    ratio = 0.8
    print(f"training 400 steps at sampling_ratio={ratio} (collapse regime)\n")

    losses: list[float] = []
    model.train()
    for step in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = model.scheduled_sampling_loss(feats, caps, ratio, generator=generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    print(f"\nloss first20 {first:.4f} -> last20 {last:.4f}")
    check("loss decreased", last < first - 0.05, f"{first:.4f} -> {last:.4f}")

    # The old failure mode: loss frozen to 6 decimal places.
    tail = losses[-100:]
    spread = max(tail) - min(tail)
    print(f"tail spread over last 100 steps: {spread:.6f}")
    check("loss is not frozen", spread > 1e-4, f"spread {spread:.6f}")

    model.eval()
    with torch.no_grad():
        logits = model(feats[:64], caps[:64])
        predictions = logits.argmax(dim=-1)
        targets = caps[:64, 1:]
        mask = targets != vocab.pad_idx
        total = int(mask.sum().item())
        counts = Counter(predictions[mask].tolist())
        pad_share = 100 * counts.get(vocab.pad_idx, 0) / max(total, 1)
        accuracy = 100 * int(((predictions == targets) & mask).sum().item()) / max(total, 1)

    print(f"\ndistinct predicted tokens: {len(counts)} | top-1 accuracy {accuracy:.2f}%")
    print(f"<pad> share: {pad_share:.1f}%")
    check("model does not predict <pad>", pad_share < 5.0, f"{pad_share:.1f}%")
    check("prediction diversity", len(counts) > 20, f"{len(counts)} distinct tokens")
    check("top-1 accuracy above chance", accuracy > 5.0, f"{accuracy:.2f}%")

    with torch.no_grad():
        shuffled = feats[torch.randperm(feats.shape[0])]
        loss_real = float(model.loss(feats[:64], caps[:64]))
        loss_shuffled = float(model.loss(shuffled[:64], caps[:64]))
    check("output depends on the image",
          abs(loss_shuffled - loss_real) > 1e-3,
          f"real {loss_real:.4f} vs shuffled {loss_shuffled:.4f}")

    failed = [c for c in checks if not c[1]]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
