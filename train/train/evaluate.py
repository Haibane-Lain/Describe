"""Evaluate a trained captioner.

Reports the official VizWiz-Captions metric set so results are comparable with
published work: BLEU-1-4, METEOR, ROUGE-L, CIDEr-D, and SPICE. Scores are
reported overall and split by the ``text_detected`` flag, because captioners
fail hardest on images containing text and that split is where the OCR branch
is supposed to help.

    py -3.12 -m train.train.evaluate ^
        --checkpoint checkpoints/vizwiz_ft/best.pt ^
        --feature-index data/features/vizwiz_test_resnet101.jsonl ^
        --manifest data/manifests/vizwiz_test.jsonl ^
        --out runs/vizwiz_ft/test_metrics.json

Falls back to a pure-Python BLEU implementation when ``pycocoevalcap`` is not
installed, and clearly marks which metrics were unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.dataset import load_feature_cache  # noqa: E402
from train.data.vocab import Vocab  # noqa: E402
from train.models.captioner import Captioner  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


# --------------------------------------------------------------- metrics
def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: list[list[str]], references: list[list[list[str]]], max_n: int = 4
) -> dict[str, float]:
    """Corpus BLEU with brevity penalty.

    A self-contained fallback so evaluation never depends on an optional
    package. Matches the standard formulation closely enough for reporting.
    """
    scores: dict[str, float] = {}
    # Drop items with no references: VizWiz includes images whose captions were
    # all rejected, so a row can legitimately have an empty reference list.
    pairs = [
        (h, refs)
        for h, refs in zip(hypotheses, references)
        if refs
    ]
    if not pairs:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "BLEU-3": 0.0, "BLEU-4": 0.0}
    hypotheses = [h for h, _ in pairs]
    references = [refs for _, refs in pairs]

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(
        min((len(r) for r in refs), key=lambda L: (abs(L - len(h)), L))
        for h, refs in zip(hypotheses, references)
    )

    for n in range(1, max_n + 1):
        match, total = 0, 0
        for hyp, refs in zip(hypotheses, references):
            hyp_ngrams = _ngrams(hyp, n)
            total += max(sum(hyp_ngrams.values()), 0)
            max_ref: Counter = Counter()
            for ref in refs:
                ref_ngrams = _ngrams(ref, n)
                for gram, count in ref_ngrams.items():
                    max_ref[gram] = max(max_ref[gram], count)
            match += sum(min(c, max_ref[gram]) for gram, c in hyp_ngrams.items())

        if total == 0 or match == 0:
            scores[f"BLEU-{n}"] = 0.0
        else:
            scores[f"BLEU-{n}"] = match / total

    if hyp_len == 0:
        scores["BLEU-4"] = 0.0
        return scores

    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0

    precisions = [scores[f"BLEU-{n}"] for n in range(1, max_n + 1)]
    if all(p > 0 for p in precisions):
        geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    else:
        geo = 0.0
    scores["BLEU-4"] = bp * geo
    return scores


def try_pycocoevalcap(
    hypotheses: list[str],
    references: list[list[str]],
    with_spice: bool = False,
    with_meteor: bool = False,
    verbose: bool = True,
) -> dict[str, float] | None:
    """Compute the VizWiz metric set with pycocoevalcap, if installed.

    Default set is BLEU-1-4, ROUGE-L, and CIDEr-D: pure-Python, fast, and
    reliable at any size. Two metrics are opt-in because they use a Java
    subprocess bridge:

    * **METEOR** (``with_meteor``) works on small inputs but hangs on large
      ones - it deadlocks at roughly 1,000+ images, apparently on the pipe
      buffer to its Java server.
    * **SPICE** (``with_spice``) needs a JDK compatible with Stanford CoreNLP
      3.6.0; on a modern JDK it fails with ``InaccessibleObjectException`` and
      can spin for minutes first.
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError:
        return None

    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}

    out: dict[str, float] = {}
    try:
        bleu, _ = Bleu(4).compute_score(gts, res)
        for n, value in enumerate(bleu, 1):
            out[f"BLEU-{n}"] = float(value)
    except Exception as exc:
        if verbose:
            print(f"  BLEU failed: {type(exc).__name__}: {exc}")

    try:
        score, _ = Rouge().compute_score(gts, res)
        out["ROUGE-L"] = float(score)
    except Exception as exc:
        if verbose:
            print(f"  ROUGE-L failed: {type(exc).__name__}: {exc}")

    try:
        score, _ = Cider().compute_score(gts, res)
        out["CIDEr-D"] = float(score)
    except Exception as exc:
        if verbose:
            print(f"  CIDEr-D failed: {type(exc).__name__}: {exc}")

    if with_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            score, _ = Meteor().compute_score(gts, res)
            out["METEOR"] = float(score)
        except Exception as exc:
            if verbose:
                print(f"  METEOR failed: {type(exc).__name__}: {exc}")

    if with_spice:
        try:
            from pycocoevalcap.spice.spice import Spice

            score, _ = Spice().compute_score(gts, res)
            out["SPICE"] = float(score)
        except Exception as exc:
            if verbose:
                print(f"  SPICE failed: {type(exc).__name__} (needs a JDK compatible with CoreNLP 3.6.0)")
    return out or None


def tokenize_simple(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[.,!?;:()\-]", text.lower())


def score_predictions(
    predictions: list[dict],
    with_spice: bool = False,
    with_meteor: bool = False,
) -> dict:
    """Compute corpus metrics and the per-`text_detected` breakdown.

    Shared by the cached-feature and end-to-end evaluation paths so both report
    the same numbers from the same code.
    """
    # Rows without references cannot be scored; exclude them everywhere.
    scorable = [
        (p["prediction"], p["references"])
        for p in predictions
        if p["references"]
    ]
    if len(scorable) < len(predictions):
        print(
            f"  note: {len(predictions) - len(scorable):,} of {len(predictions):,} "
            "images have no references (all captions rejected) and are excluded"
        )
    hypotheses = [h for h, _ in scorable]
    references = [refs for _, refs in scorable]

    metrics = try_pycocoevalcap(
        hypotheses, references, with_spice=with_spice, with_meteor=with_meteor
    )
    if metrics is None:
        print("  pycocoevalcap unavailable - reporting BLEU only")
        metrics = corpus_bleu(
            [tokenize_simple(h) for h in hypotheses],
            [[tokenize_simple(r) for r in refs] for refs in references],
        )

    split_metrics: dict[str, dict] = {}
    for label, predicate in (
        ("text_detected_true", lambda v: v is True),
        ("text_detected_false", lambda v: v is False),
        ("text_detected_unknown", lambda v: v is None),
    ):
        subset = [
            p for p in predictions
            if predicate(p["text_detected"]) and p["references"]
        ]
        if len(subset) < 5:
            continue
        sub_hyp = [p["prediction"] for p in subset]
        sub_ref = [p["references"] for p in subset]
        sub = try_pycocoevalcap(
            sub_hyp, sub_ref,
            with_spice=with_spice, with_meteor=with_meteor, verbose=False,
        )
        if sub is None:
            sub = corpus_bleu(
                [tokenize_simple(h) for h in sub_hyp],
                [[tokenize_simple(r) for r in refs] for refs in sub_ref],
            )
        split_metrics[label] = {"n": len(subset), **sub}

    return {"metrics": metrics, "by_text_detected": split_metrics}


def evaluate_end_to_end(
    model,
    rows: list[dict],
    vocab,
    config: dict,
    image_root: Path,
    device,
    batch_size: int = 32,
    beam_size: int = 5,
    max_len: int = 30,
    min_len: int = 2,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    length_alpha: float = 0.7,
    with_spice: bool = False,
    with_meteor: bool = False,
    payload_epoch=None,
    checkpoint: str = "",
) -> tuple[dict, list[dict]]:
    """Evaluate a checkpoint whose encoder is trained (needs raw images)."""
    from PIL import Image

    from train.features.extract import preprocess

    image_size = int(config.get("image_size", 224))
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[path.name] = path

    predictions: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tensors = []
        for row in chunk:
            path = index.get(row["file_name"])
            if path is None:
                raise SystemExit(f"image not found for {row['file_name']} under {image_root}")
            with Image.open(path) as handle:
                tensors.append(preprocess(handle, image_size))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            token_seqs = model.decode(
                batch, vocab.start_idx, vocab.end_idx,
                beam_size=beam_size, max_len=max_len, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram, length_alpha=length_alpha,
            )
        for row, tokens in zip(chunk, token_seqs):
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "text_detected": row.get("text_detected"),
                    "prediction": vocab.decode(tokens),
                    "references": row.get("captions", []),
                }
            )
        done = min(start + batch_size, len(rows))
        print(f"\r  decoded {done:,}/{len(rows):,}", end="", flush=True)
    print()

    scored = score_predictions(predictions, with_spice, with_meteor)
    result = {
        "checkpoint": str(checkpoint),
        "epoch": payload_epoch,
        "beam_size": beam_size,
        "n_images": len(predictions),
        "end_to_end": True,
        **scored,
    }
    return result, predictions


# ---------------------------------------------------------------- evaluation
def evaluate(
    checkpoint: Path,
    feature_index: Path,
    manifest: Path,
    beam_size: int = 5,
    max_len: int = 30,
    min_len: int = 5,
    batch_size: int = 32,
    limit: int | None = None,
    cpu: bool = False,
    with_spice: bool = False,
    with_meteor: bool = False,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> dict:
    info = get_device(prefer_gpu=not cpu)
    configure_for_speed()
    print(f"Device    : {info.name} ({info.backend})")

    payload = torch.load(checkpoint, map_location=info.device, weights_only=False)
    config = payload.get("config", {})

    vocab_path = checkpoint.parent / "vocab.pkl"
    if not vocab_path.exists():
        raise SystemExit(f"vocab not found next to checkpoint: {vocab_path}")
    vocab = Vocab.load(vocab_path)

    model_type = config.get("model", "lstm")
    end_to_end = bool(config.get("end_to_end"))
    if model_type == "attention":
        from train.models.attention_captioner import AttentionCaptioner

        decoder = AttentionCaptioner(
            vocab_size=len(vocab),
            feature_dim=config.get("feature_dim", 2048),
            embed_dim=config.get("embed_dim", 512),
            hidden_dim=config.get("hidden_dim", 512),
            attn_dim=config.get("attn_dim", 512),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.5),
            pad_idx=vocab.pad_idx,
        ).to(info.device)
        model = decoder
    else:
        model = Captioner(
            vocab_size=len(vocab),
            feature_dim=config.get("feature_dim", 2048),
            embed_dim=config.get("embed_dim", 512),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.5),
            pad_idx=vocab.pad_idx,
        ).to(info.device)

    state = payload["model_state"]
    if end_to_end:
        # End-to-end checkpoints wrap encoder + decoder, so the decoder keys are
        # prefixed and the encoder weights live under `encoder.*`. Rebuild the
        # wrapper and load it whole.
        from train.features.extract import build_spatial_encoder
        from train.train.train_e2e import EncoderDecoder

        encoder, _ = build_spatial_encoder(
            config.get("backbone", "resnet50"), int(config.get("image_size", 224))
        )
        model = EncoderDecoder(decoder, encoder.to(info.device)).to(info.device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model     : {model_type}{' (end-to-end)' if end_to_end else ''}")
    print(f"Checkpoint: {checkpoint} (epoch {payload.get('epoch', '?')})")

    # End-to-end checkpoints consume raw images and have no feature cache, so
    # this branch must come before any cache loading. It previously sat after,
    # which made every end-to-end evaluation fail on a missing .npy.
    if end_to_end:
        manifest_rows = [
            json.loads(line) for line in manifest.open(encoding="utf-8")
            if line.strip()
        ]
        if limit:
            manifest_rows = manifest_rows[:limit]
        print(f"Images    : {len(manifest_rows):,}")
        return evaluate_end_to_end(
            model=model, rows=manifest_rows, vocab=vocab, config=config,
            image_root=Path(config.get("image_root", "data/raw/flickr30k/images")),
            device=info.device, batch_size=batch_size, beam_size=beam_size,
            max_len=max_len, min_len=min_len,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
            with_spice=with_spice, with_meteor=with_meteor,
            payload_epoch=payload.get("epoch"), checkpoint=str(checkpoint),
        )

    # Spatial maps are read lazily; a 6 GB array must not be pulled into RAM.
    features, cached_rows = load_feature_cache(
        feature_index, mmap=(model_type == "attention")
    )

    # The cache index is authoritative and carries `feature_row`.
    manifest_rows = [
        json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()
    ]
    requested = {row["image_id"] for row in manifest_rows}
    rows = [row for row in cached_rows if row["image_id"] in requested]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(
            f"no evaluable rows: nothing in {manifest} is present in {feature_index}"
        )

    print(f"Images    : {len(rows):,}")

    predictions: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        row_indices = [r["feature_row"] for r in chunk]
        batch_features = torch.from_numpy(
            features[row_indices].astype("float32")
        ).to(info.device)

        if beam_size > 1:
            token_seqs = model.beam_decode(
                batch_features, vocab.start_idx, vocab.end_idx,
                beam_size=beam_size, max_len=max_len, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )
        else:
            token_seqs = model.greedy_decode(
                batch_features, vocab.start_idx, vocab.end_idx,
                max_len=max_len, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )

        for row, tokens in zip(chunk, token_seqs):
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "text_detected": row.get("text_detected"),
                    "prediction": vocab.decode(tokens),
                    "references": row.get("captions", []),
                }
            )
        done = min(start + batch_size, len(rows))
        print(f"\r  decoded {done:,}/{len(rows):,}", end="", flush=True)
    print()

    scored = score_predictions(predictions, with_spice, with_meteor)
    result = {
        "checkpoint": str(checkpoint),
        "epoch": payload.get("epoch"),
        "beam_size": beam_size,
        "n_images": len(predictions),
        **scored,
    }
    return result, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained captioner")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write metrics JSON here")
    parser.add_argument("--pred-out", type=Path, help="write per-image predictions JSONL")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=30)
    # VizWiz captions average ~10-12 tokens; a large min_len forces unnatural
    # padding, so keep it well below the expected length.
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--with-meteor",
        action="store_true",
        help="also attempt METEOR; works on small sets but hangs above ~1000 images",
    )
    parser.add_argument(
        "--with-spice",
        action="store_true",
        help="also attempt SPICE; needs a JDK compatible with CoreNLP 3.6.0 "
             "(fails and can spin for minutes on modern JDKs)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="suppress already-emitted tokens (>1 enables; 1.2-1.3 typical)",
    )
    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=0,
        help="forbid repeating an n-gram of this size (3 blocks 'a a a')",
    )
    args = parser.parse_args()

    result, predictions = evaluate(
        checkpoint=args.checkpoint,
        feature_index=args.feature_index,
        manifest=args.manifest,
        beam_size=args.beam_size,
        max_len=args.max_len,
        min_len=args.min_len,
        batch_size=args.batch_size,
        limit=args.limit,
        cpu=args.cpu,
        with_spice=args.with_spice,
        with_meteor=args.with_meteor,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
    )

    print("\nMetrics")
    for name, value in sorted(result["metrics"].items()):
        print(f"  {name:10s} {value:.4f}")
    for label, values in result["by_text_detected"].items():
        print(f"\n  {label} (n={values['n']})")
        for name, value in sorted(values.items()):
            if name == "n":
                continue
            print(f"    {name:10s} {value:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    if args.pred_out:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_out.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction) + "\n")
        print(f"Wrote {args.pred_out} ({len(predictions):,} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
