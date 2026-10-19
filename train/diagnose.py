"""Decide whether the captioner's problem is vision or language.

Three measurements, because they lead to very different fixes:

1. **Image-use gap** — cross-entropy on correct vs shuffled vs zero images. A
   small gap means the decoder is running on the language prior and no decoder
   swap will help.
2. **Repetition profile** — how often the model repeats the previous token, and
   how many distinct tokens it emits. This quantifies the filler directly.
3. **Decoding vs generation** — greedy against beam. If greedy is fluent and
   beam is not, the bug is in decoding and costs hours, not a day.

Works with both the cached-feature and end-to-end checkpoints.

    py -3.12 -m train.diagnose --checkpoint checkpoints/flickr30k_e2e_safe/best.pt
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


def build_model(checkpoint: Path, vocab: Vocab, config: dict, device):
    """Rebuild the checkpoint's model, end-to-end or cached-feature."""
    end_to_end = bool(config.get("end_to_end"))
    decoder = AttentionCaptioner(
        vocab_size=len(vocab),
        feature_dim=config.get("feature_dim", 2048),
        embed_dim=config.get("embed_dim", 512),
        hidden_dim=config.get("hidden_dim", 512),
        attn_dim=config.get("attn_dim", 512),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.5),
        pad_idx=vocab.pad_idx,
    )
    if not end_to_end:
        return decoder.to(device), False
    from train.features.extract import build_spatial_encoder
    from train.train.train_e2e import EncoderDecoder

    encoder, _ = build_spatial_encoder(
        config.get("backbone", "resnet50"), int(config.get("image_size", 224))
    )
    return EncoderDecoder(decoder, encoder).to(device), True


def load_images(rows, config, limit):
    """Return [N, C, H, W] tensors plus captions, for either model kind."""
    from PIL import Image

    from train.features.extract import preprocess

    end_to_end = bool(config.get("end_to_end"))
    if end_to_end:
        root = Path(config.get("image_root", "data/raw/flickr30k/images"))
        index = {}
        for path in root.rglob("*"):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                index[path.name] = path
        size = int(config.get("image_size", 224))
        tensors = []
        for row in rows[:limit]:
            with Image.open(index[row["file_name"]]) as handle:
                tensors.append(preprocess(handle, size))
        return torch.stack(tensors)

    index_path = Path("data/features/flickr30k_resnet101_spatial.jsonl")
    features, _ = load_feature_cache(index_path, mmap=True)
    indices = [r["feature_row"] for r in rows[:limit]]
    return torch.from_numpy(np.asarray(features[indices], dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description="Vision vs language diagnostic")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/flickr30k_holdout.jsonl"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    configure_for_speed()
    info = get_device(prefer_gpu=not args.cpu)
    device = info.device
    print(f"device: {info.name}\n")

    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = payload.get("config", {})
    vocab = Vocab.load(args.checkpoint.parent / "vocab.pkl")
    model, end_to_end = build_model(args.checkpoint, vocab, config, device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print(f"checkpoint: {args.checkpoint} (epoch {payload.get('epoch')})")
    print(f"kind      : {'end-to-end' if end_to_end else 'cached features'}")
    print(f"vocab     : {len(vocab):,}\n")

    rows = [
        json.loads(line) for line in args.manifest.open(encoding="utf-8")
        if line.strip()
    ]
    rows = [r for r in rows if r.get("captions")][: args.limit]
    inputs = load_images(rows, config, args.limit).to(device)
    captions = collate_captions(
        [vocab.encode(r["captions"][0]) for r in rows], vocab.pad_idx
    ).to(device)

    def loss_on(x: torch.Tensor) -> float:
        with torch.no_grad():
            if end_to_end:
                return float(model.loss(x, captions))
            return float(model.decoder.loss(x, captions))

    # ---------------------------------------------------- 1. image-use gap
    real = loss_on(inputs)
    shuffled = loss_on(inputs[torch.randperm(inputs.shape[0])])
    zeros = loss_on(torch.zeros_like(inputs))
    print("1. Image-use gap (cross-entropy, lower = better)")
    print(f"   correct image : {real:.4f}")
    print(f"   shuffled      : {shuffled:.4f}   (delta {shuffled - real:+.4f})")
    print(f"   zero image    : {zeros:.4f}   (delta {zeros - real:+.4f})")
    gap = shuffled - real
    uses_image = gap > 0.1
    print(f"   -> {'USES the image' if uses_image else 'IGNORES the image'}")

    # ------------------------------------------------- 2. repetition profile
    with torch.no_grad():
        if end_to_end:
            features = model.encoder(inputs)
            logits = model.decoder(features, captions)
        else:
            logits = model(inputs, captions)
        predictions = logits.argmax(dim=-1)
        targets = captions[:, 1:]
        mask = targets != vocab.pad_idx
        total = int(mask.sum().item())
        counts = Counter(predictions[mask].tolist())
        # Repetition: does position t predict the same token as position t-1?
        same = (predictions[:, 1:] == predictions[:, :-1]) & mask[:, 1:]
        repeat_rate = float(same.sum().item()) / max(int(mask[:, 1:].sum().item()), 1)
        top_share = 100 * counts.most_common(1)[0][1] / max(total, 1)
        top_token = vocab.idx2word[counts.most_common(1)[0][0]]

    print("\n2. Prediction profile (teacher-forced)")
    print(f"   distinct tokens : {len(counts):,} of {len(vocab):,}")
    print(f"   most common     : {top_token!r} at {top_share:.1f}%")
    print(f"   repeat-prev rate: {repeat_rate:.3f}")
    fluent = repeat_rate < 0.25
    print(f"   -> {'fluent-ish' if fluent else 'REPETITIVE (language-model failure)'}")

    # --------------------------------------------- 3. greedy vs beam decode
    beam = model.decode(
        inputs[:8], vocab.start_idx, vocab.end_idx,
        beam_size=5, max_len=30, min_len=2,
        repetition_penalty=1.2, no_repeat_ngram=3, length_alpha=0.7,
    ) if end_to_end else model.beam_decode(
        inputs[:8], vocab.start_idx, vocab.end_idx,
        beam_size=5, max_len=30, min_len=2,
        repetition_penalty=1.2, no_repeat_ngram=3, length_alpha=0.7,
    )
    greedy = model.decoder.greedy_decode(
        model.encoder(inputs[:8]) if end_to_end else inputs[:8],
        vocab.start_idx, vocab.end_idx,
        max_len=30, min_len=2, repetition_penalty=1.2, no_repeat_ngram=3,
    )
    raw_beam = model.decoder.beam_decode(
        model.encoder(inputs[:8]) if end_to_end else inputs[:8],
        vocab.start_idx, vocab.end_idx,
        beam_size=5, max_len=30, min_len=2, length_alpha=0.0,
    )

    def repeat_of(seqs) -> float:
        hits = sum(
            1 for seq in seqs for a, b in zip(seq, seq[1:]) if a == b
        )
        pairs = sum(max(len(seq) - 1, 0) for seq in seqs)
        return hits / max(pairs, 1)

    print("\n3. Decoding comparison (repeat-adjacent rate, lower = better)")
    print(f"   greedy (pen 1.2)   : {repeat_of(greedy):.3f}")
    print(f"   beam-5 (pen 1.2)   : {repeat_of(beam):.3f}")
    print(f"   beam-5 (no penalty): {repeat_of(raw_beam):.3f}")
    print("\n   samples")
    for i in range(min(3, len(greedy))):
        print(f"     greedy: {vocab.decode(greedy[i])!r}")
        print(f"     beam  : {vocab.decode(beam[i])!r}")
        print(f"     ref   : {rows[i]['captions'][0]!r}")
        print()

    # ------------------------------------------------------------- verdict
    print("Verdict")
    if not uses_image:
        print("  Decoder IGNORES the image -> fix conditioning, not the decoder.")
    elif not fluent:
        print("  Image is used but text is repetitive -> LANGUAGE-MODEL failure.")
        print("  A transformer decoder targets this; more vision work will not.")
    else:
        print("  Both signals healthy -> look at data volume and training length.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
