"""Check whether an attention checkpoint has collapsed.

Loads a checkpoint and runs the same batch repeatedly, reporting whether the
loss is genuinely constant (collapse) or merely varying in a narrow band
(plateau), plus how many distinct tokens the model predicts.

    py -3.12 -m train.quality... # (not this module)
    py -3.12 -m train.check_attn --checkpoint checkpoints/flickr30k_attn/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data.dataset import load_feature_cache  # noqa: E402
from train.data.vocab import Vocab  # noqa: E402
from train.models.attention_captioner import AttentionCaptioner  # noqa: E402
from train.models.captioner import collate_captions  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/flickr30k.jsonl"))
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    info = get_device(prefer_gpu=not args.cpu)
    configure_for_speed()
    print(f"device: {info.name}\n")

    payload = torch.load(args.checkpoint, map_location=info.device, weights_only=False)
    config = payload.get("config", {})
    vocab = Vocab.load(args.checkpoint.parent / "vocab.pkl")
    print(f"checkpoint epoch {payload.get('epoch')}  vocab {len(vocab):,}")

    model = AttentionCaptioner(
        vocab_size=len(vocab),
        feature_dim=config.get("feature_dim", 2048),
        embed_dim=config.get("embed_dim", 512),
        hidden_dim=config.get("hidden_dim", 512),
        attn_dim=config.get("attn_dim", 512),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.5),
        pad_idx=vocab.pad_idx,
    ).to(info.device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    index = Path("data/features") / f"{args.manifest.stem}_resnet101_spatial.jsonl"
    features, cached_rows = load_feature_cache(index, mmap=True)
    rows = [r for r in cached_rows if r.get("captions")][: args.batch]
    print(f"using {len(rows)} images from {index.name}\n")

    indices = [r["feature_row"] for r in rows]
    feats = torch.from_numpy(
        np.asarray(features[indices], dtype=np.float32)
    ).to(info.device)
    caps = collate_captions(
        [vocab.encode(r["captions"][0]) for r in rows], vocab.pad_idx
    ).to(info.device)

    # 1. Is the loss constant under repeated identical input?
    losses = []
    with torch.no_grad():
        for _ in range(5):
            losses.append(float(model.loss(feats, caps)))
    spread = max(losses) - min(losses)
    print("Repeated identical batch")
    print(f"  losses      : {[round(v, 6) for v in losses]}")
    print(f"  spread      : {spread:.8f}")

    # 2. Does the loss change when the images change?
    shuffled = feats[torch.randperm(feats.shape[0])]
    with torch.no_grad():
        loss_shuffled = float(model.loss(shuffled, caps))
        zeros = float(model.loss(torch.zeros_like(feats), caps))
    print("\nSensitivity to the image")
    print(f"  correct image : {losses[0]:.4f}")
    print(f"  shuffled image: {loss_shuffled:.4f}  (delta {loss_shuffled - losses[0]:+.4f})")
    print(f"  zero image    : {zeros:.4f}  (delta {zeros - losses[0]:+.4f})")

    # 3. How many distinct tokens does it predict, and is `a` dominating?
    with torch.no_grad():
        logits = model(feats, caps)
        predictions = logits.argmax(dim=-1)
        targets = caps[:, 1:]
        mask = targets != vocab.pad_idx
        total = int(mask.sum().item())
        correct = int(((predictions == targets) & mask).sum().item())
        token_counts = Counter(predictions[mask].tolist())

    print("\nPrediction diversity")
    print(f"  top-1 accuracy : {100 * correct / max(total, 1):.2f}%")
    print(f"  distinct tokens: {len(token_counts)} of {len(vocab):,}")
    for idx, count in token_counts.most_common(5):
        share = 100 * count / max(total, 1)
        print(f"    {vocab.idx2word[idx]!r:14s} {share:5.1f}%")

    # 4. Does attention actually look at different places for different images?
    #    If the queries are image-blind the weight maps collapse to the same
    #    distribution regardless of input, and the model runs on the language
    #    prior alone. This was a real bug: the attention queries came from an
    #    LSTM pass fed a zero context.
    with torch.no_grad():
        flat = model.flatten_features(feats[:8])
        keys = model.attention.feature_proj(flat)                # [B, L, A]
        hidden, cell = model._init_hidden(flat, flat.shape[0])
        seed = model.context_proj(flat.mean(dim=1))
        embedded = model.embed(caps[:8])
        tokens_in = torch.cat([seed.unsqueeze(1), embedded[:, :-1, :]], dim=1)
        outputs, _ = model.lstm(
            torch.cat([tokens_in, torch.zeros_like(tokens_in)], dim=-1),
            (hidden, cell),
        )
        queries = torch.cat([hidden[-1].unsqueeze(1), outputs[:, :-1, :]], dim=1)
        energies = model.attention.score(
            torch.tanh(
                model.attention.hidden_proj(queries).unsqueeze(2) + keys.unsqueeze(1)
            )
        ).squeeze(-1)
        weights = torch.softmax(energies, dim=-1)                # [B, T, L]
        reference = weights[0].mean(dim=0)                       # [L]
        divergences = [
            float((weights[i].mean(dim=0) - reference).abs().sum())
            for i in range(1, weights.shape[0])
        ]
        mean_divergence = sum(divergences) / max(len(divergences), 1)
        entropy = float(-(weights * (weights + 1e-9).log()).sum(dim=-1).mean())
        max_entropy = float(torch.log(torch.tensor(float(weights.shape[-1]))))

    print("\nAttention behaviour")
    print(f"  mean |delta attention| vs image 0 : {mean_divergence:.4f}")
    print(f"  attention entropy                 : {entropy:.4f} "
          f"(uniform = {max_entropy:.4f})")
    attention_ok = mean_divergence > 1e-4 and entropy < max_entropy * 0.999
    print(f"  [{'PASS' if mean_divergence > 1e-4 else 'FAIL'}] attention differs across images")
    print(f"  [{'PASS' if entropy < max_entropy * 0.999 else 'FAIL'}] attention is not uniform")

    # 5. Free-running output
    with torch.no_grad():
        decoded = model.beam_decode(
            feats[:3], vocab.start_idx, vocab.end_idx,
            beam_size=5, max_len=30, min_len=2,
            repetition_penalty=1.2, no_repeat_ngram=3,
        )
    print("\nBeam-5 samples")
    for i, tokens in enumerate(decoded):
        print(f"  PRED: {vocab.decode(tokens)!r}")
        print(f"  REF : {rows[i]['captions'][0]!r}")

    collapsed = spread < 1e-6 and abs(loss_shuffled - losses[0]) < 1e-4
    print(f"\n  verdict: {'COLLAPSED - output is independent of input' if collapsed else 'not collapsed'}")
    return 0 if attention_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
