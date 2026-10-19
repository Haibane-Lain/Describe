"""Batched beam-search evaluation for a cached-feature BLIP captioner.

Two things this file deliberately does not do:

**It does not re-implement the metrics.** Scoring goes through
``train.train.evaluate.score_predictions``, the same function the from-scratch
captioner's evaluation uses. A BLIP number and an LSTM number that came from
different scoring code are not comparable, and the whole point of this work is
the comparison.

**It does not decode one image at a time.** The from-scratch beam loop decoded
serially (see ``train/README.md``, "Known limitations"), which is why a
7,750-image evaluation was a scheduling concern. Here generation is batched over
cached features, so a full validation pass is minutes.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.base import (  # noqa: E402
    build_model,
    load_trainable_state_dict,
    prepare_model_for_checkpoint,
    resolve_model_id,
)
from train.finetune.cache import CachePaths, load_cache, row_image_key  # noqa: E402
from train.finetune.dataset import build_split, captions_of, load_index  # noqa: E402
from train.train.evaluate import tokenize_simple  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


# --------------------------------------------------------------- generation


@torch.no_grad()
def decode_cached(
    model,
    index: list[dict],
    features: np.ndarray,
    tokenizer,
    device,
    *,
    batch_size: int = 64,
    num_beams: int = 5,
    max_new_tokens: int = 30,
    min_new_tokens: int = 2,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    limit: int | None = None,
    progress_every: int = 25,
    bf16: bool = True,
) -> list[str]:
    """Beam-search captions for every cached image, in batches.

    Starts from the same token sequence ``BlipForConditionalGeneration.generate``
    uses - ``[bos, sep]``, with ``sep`` as the EOS - so the decoding contract is
    identical to the reference implementation. That equivalence is asserted in
    ``smoke_test_finetune.py``.

    ``bf16`` is not a micro-optimisation here, it is the difference between a
    usable and an unusable evaluation loop. Measured on this GPU, beam-5 over 64
    images:

    | dtype | use_cache | img/s |
    |---|---|---|
    | fp32 | True | 2.3 |
    | **bf16** | **True** | **8.9** |
    | fp32 | False | 1.7 |
    | bf16 | True, beam-1 | 147.5 |

    At fp32 a full 7,750-image beam-5 pass takes ~54 minutes, which makes
    per-epoch beam-5 validation impossible. At bf16 it is ~15 minutes, and
    greedy decoding is fast enough (~150 img/s) to use for per-epoch
    checkpoint selection. ``use_cache`` is left on; turning it off costs 35%.
    """
    text_config = model.config.text_config
    decoder_start = int(text_config.bos_token_id)
    eos_token_id = int(getattr(text_config, "sep_token_id", None) or text_config.eos_token_id)
    pad_token_id = int(text_config.pad_token_id)

    rows = index[:limit] if limit else index
    # Inference must run in eval mode. `generate()` does not switch the module
    # itself, so a caller that forgot would decode with dropout active and get
    # degraded, non-reproducible captions. That is exactly what happened here:
    # the cached path and the live path disagreed on 7 of 20 real images, and
    # the difference was dropout in the cached path. BLIP's own config sets both
    # dropout probabilities to 0.0, so nothing in the project's normal runs
    # surfaces it - which is precisely why it needs to be explicit.
    model.eval()

    # Set the token ids on the generation config rather than passing them as
    # kwargs: a modern `generate()` binds eos/pad from the config itself, so
    # passing them again raises "got multiple values for keyword argument".
    decoder = model.text_decoder
    decoder.generation_config.eos_token_id = eos_token_id
    decoder.generation_config.pad_token_id = pad_token_id
    decoder.generation_config.bos_token_id = decoder_start

    use_bf16 = bf16 and device.type == "cuda"
    captions: list[str] = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        row_ids = [int(row["feature_row"]) for row in chunk]
        embeds = torch.from_numpy(np.asarray(features[row_ids], dtype=np.float32)).to(device)

        start_ids = torch.full((len(chunk), 1), decoder_start, dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            output = decoder.generate(
                input_ids=start_ids,
                encoder_hidden_states=embeds,
                encoder_attention_mask=torch.ones(embeds.shape[:-1], dtype=torch.long, device=device),
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

        # generate() returns only the new tokens when input_ids are supplied;
        # guard both shapes rather than assuming one.
        for row_tokens in output:
            ids = row_tokens.tolist()
            if ids and ids[0] == decoder_start:
                ids = ids[1:]
            captions.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        done = min(start + batch_size, len(rows))
        if (start // batch_size) % progress_every == 0 or done == len(rows):
            print(f"\r  decoded {done:,}/{len(rows):,}", end="", flush=True)

    print()
    return captions


# --------------------------------------------------------------- diagnostics


def collapse_diagnostics(predictions: list[dict]) -> dict:
    """Vocabulary-collapse statistics for a set of predictions.

    This is the measurement that the from-scratch project should have been
    reporting all along. Adjacent-repeat rate was the metric used to declare a
    model healthy; the collapsed model passed that test trivially (0.047-0.085)
    while emitting only 126 of 7,488 tokens. These three numbers would not have
    passed:

    * ``distinct_tokens`` - how much of the language the model can actually use.
    * ``top_token_share`` - whether output is a high-frequency unigram bag.
    * ``ngram_match_rate`` - whether any produced 4-gram appears in any human
      reference. At 0.0 the model is not generating phrases at all.
    """
    token_lists = [tokenize_simple(p["prediction"]) for p in predictions]
    flat = [token for tokens in token_lists for token in tokens]
    counts = Counter(flat)

    scorable = [p for p in predictions if p["references"]]

    def ngram_match(n: int) -> float:
        hits = 0
        total = 0
        for record in scorable:
            tokens = tokenize_simple(record["prediction"])
            if len(tokens) < n:
                continue
            total += 1
            produced = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
            reference: set[tuple[str, ...]] = set()
            for ref in record["references"]:
                ref_tokens = tokenize_simple(ref)
                reference |= {
                    tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
                }
            if produced & reference:
                hits += 1
        return hits / total if total else 0.0

    reference_tokens: Counter = Counter()
    for record in scorable:
        for ref in record["references"]:
            reference_tokens.update(tokenize_simple(ref))

    lengths = [len(tokens) for tokens in token_lists]
    top_token, top_count = counts.most_common(1)[0] if counts else ("", 0)

    return {
        "n_images": len(predictions),
        "distinct_tokens": len(counts),
        "total_tokens": len(flat),
        "top_token": top_token,
        "top_token_share": top_count / len(flat) if flat else 0.0,
        "top10_share": sum(c for _, c in counts.most_common(10)) / len(flat) if flat else 0.0,
        "reference_vocab_size": len(reference_tokens),
        "reference_mass_covered": (
            sum(counts[t] for t in counts if t in reference_tokens) / len(flat) if flat else 0.0
        ),
        "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "empty_predictions": sum(1 for tokens in token_lists if not tokens),
        "bigram_match_rate": ngram_match(2),
        "fourgram_match_rate": ngram_match(4),
        "most_common_predicted": [
            {"token": token, "predicted_pct": 100 * count / len(flat)}
            for token, count in counts.most_common(10)
        ]
        if flat
        else [],
        "most_common_reference": [
            {"token": token, "reference_pct": 100 * count / sum(reference_tokens.values())}
            for token, count in reference_tokens.most_common(10)
        ]
        if reference_tokens
        else [],
    }


# --------------------------------------------------------------- entry point


def evaluate_predictions(
    predictions: list[dict],
    *,
    with_meteor: bool = False,
    with_spice: bool = False,
) -> dict:
    """Score predictions through the shared metric implementation."""
    from train.train.evaluate import score_predictions

    scored = score_predictions(predictions, with_spice=with_spice, with_meteor=with_meteor)
    return {**scored, "diagnostics": collapse_diagnostics(predictions)}


def load_run_adapter(
    model,
    run_dir: Path,
    checkpoint: Path | None = None,
    *,
    zero_shot: bool = False,
) -> tuple[list[str], list[str]]:
    """Load a run's checkpoint into the model.

    Checkpoints live under ``checkpoints/<run-name>/`` while logs and metrics
    live under ``runs/<run-name>/`` - the split this repo already uses - so
    accepting only a run directory would look in the wrong place. Search both,
    and allow an explicit path.

    With ``zero_shot=True`` nothing is loaded, which is the **baseline this
    project needs**: BLIP is already trained on COCO, so its unadapted score is
    the number every "fine-tuning helped" claim has to beat. Measured on 500
    VizWiz val images it reaches BLEU-4 0.168 with no training at all.
    """
    if zero_shot:
        print("checkpoint : none (zero-shot, unadapted pretrained model)")
        return [], []
    candidates: list[Path] = []
    if checkpoint is not None:
        candidates.append(checkpoint)
    else:
        candidates.extend([run_dir / "best.pt", run_dir / "last.pt"])
        # `runs/<name>` -> `checkpoints/<name>`
        candidates.extend(
            [
                Path("checkpoints") / run_dir.name / "best.pt",
                Path("checkpoints") / run_dir.name / "last.pt",
            ]
        )

    resolved = next((path for path in candidates if path.exists()), None)
    if resolved is None:
        tried = "\n".join(f"    {path}" for path in candidates)
        raise SystemExit(f"no checkpoint found. Tried:\n{tried}")

    print(f"checkpoint : {resolved}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    prepare_model_for_checkpoint(model, payload)

    loaded, skipped = load_trainable_state_dict(model, state)
    if not loaded:
        raise SystemExit(
            f"loaded 0 of {len(state)} tensors from {resolved}. The checkpoint does not match this "
            "model. Refusing to continue: reporting the pretrained model's metrics under a "
            "fine-tuned run's name would be a wrong result that looks entirely plausible."
        )
    return loaded, skipped


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a cached-feature BLIP captioner.")
    parser.add_argument("--run-dir", type=Path, required=True, help="run directory holding history and metrics")
    parser.add_argument("--zero-shot", action="store_true",
                        help="score the unadapted pretrained model; no checkpoint is loaded")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit checkpoint; default searches runs/<name> and checkpoints/<name>")
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-tensor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="restrict to rows from this manifest")
    parser.add_argument("--model", default="base", help="'base', 'large', or a Hub repo id")
    parser.add_argument("--out", type=Path, help="metrics JSON")
    parser.add_argument("--pred-out", type=Path, help="per-image predictions JSONL")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--with-meteor", action="store_true")
    parser.add_argument("--with-spice", action="store_true")
    parser.add_argument("--examples", type=int, default=12, help="print this many prediction/reference pairs")
    parser.add_argument("--no-bf16", action="store_true",
                        help="disable bf16 autocast during generation (about 4x slower)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    configure_for_speed()
    info = get_device()
    device = torch.device("cpu") if args.cpu else info.device
    print(f"device     : {info.name} ({info.backend})" if not args.cpu else "device     : cpu")

    features, index = load_cache(CachePaths(index=args.cache_index, tensor=args.cache_tensor))
    rows = build_split(index)
    if args.manifest is not None:
        wanted = {
            (row.get("image") or row.get("file"))
            for row in (json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip())
        }
        rows = [row for row in rows if (row.get("image") or row.get("file")) in wanted]
        print(f"manifest   : {args.manifest} -> {len(rows)} rows of the cache")
    print(f"rows       : {len(rows)} with captions" + (f" (limited to {args.limit})" if args.limit else ""))

    from transformers import AutoTokenizer

    model_id = resolve_model_id(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model, report = build_model(model_id, with_vision=False, dtype=torch.float32, device=device)
    if report.num_image_tokens != features.shape[1]:
        raise SystemExit(
            f"cache was built for {features.shape[1]} image tokens but {model_id} expects "
            f"{report.num_image_tokens}; re-run train.finetune.cache with the right --model"
        )
    # The text decoder's cross-attention must know the vision hidden size, which
    # the wrapper takes from config - nothing to fix up here.

    loaded, skipped = load_run_adapter(model, args.run_dir, args.checkpoint, zero_shot=args.zero_shot)
    if not args.zero_shot:
        print(f"loaded     : {len(loaded)} tensors" + (f", skipped {len(skipped)}" if skipped else ""))

    captions = decode_cached(
        model, rows, features, tokenizer, device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        limit=args.limit,
        bf16=not args.no_bf16,
    )

    predictions = []
    for row, caption in zip(rows[: len(captions)], captions):
        predictions.append(
            {
                "image_id": row.get("image_id") or row.get("id"),
                # Use the same resolver the cache uses. Manifests here identify
                # images by `file_name`, not `image`, and getting this wrong
                # produced a prediction file whose filenames matched nothing -
                # so the quality/flaw join silently split 0 rows and reported
                # only the aggregate. The mismatch is invisible unless the join
                # coverage is printed, which is why it is now printed.
                "file_name": row_image_key(row) or row.get("image") or row.get("file"),
                "text_detected": row.get("text_detected"),
                "prediction": caption,
                "references": captions_of(row),
            }
        )

    result = {
        "run_dir": str(args.run_dir),
        "model_id": model_id,
        "n_images": len(predictions),
        "beam_size": args.num_beams,
        **evaluate_predictions(
            predictions, with_meteor=args.with_meteor, with_spice=args.with_spice
        ),
    }

    if args.pred_out:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_out.open("w", encoding="utf-8") as handle:
            for record in predictions:
                handle.write(json.dumps(record) + "\n")
        print(f"predictions: {args.pred_out}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"metrics    : {args.out}")

    _print_summary(result, predictions, args.examples)
    return 0


def _print_summary(result: dict, predictions: list[dict], examples: int) -> None:
    metrics = result.get("metrics", {})
    print("\n  metric            value")
    for key in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"):
        if key in metrics:
            print(f"  {key:<16} {metrics[key]:.4f}")

    diag = result.get("diagnostics", {})
    if diag:
        print("\n  collapse diagnostics (the from-scratch model scored: 126 distinct tokens, "
              "top token 'a' at 26.9%, 0.0% 4-gram match)")
        print(f"  distinct_tokens     {diag['distinct_tokens']}  (of {diag['reference_vocab_size']} in references)")
        print(f"  top token           {diag['top_token']!r} at {100 * diag['top_token_share']:.1f}%")
        print(f"  top-10 share        {100 * diag['top10_share']:.1f}%")
        print(f"  reference mass      {100 * diag['reference_mass_covered']:.1f}%")
        print(f"  bigram match rate   {100 * diag['bigram_match_rate']:.1f}%")
        print(f"  4-gram match rate   {100 * diag['fourgram_match_rate']:.1f}%")
        print(f"  mean length         {diag['mean_length']:.1f} tokens")
        print(f"  empty predictions   {diag['empty_predictions']}")

    by_text = result.get("by_text_detected", {})
    if by_text:
        print("\n  split                      n      BLEU-1   BLEU-4   CIDEr-D")
        for label, values in sorted(by_text.items()):
            print(
                f"  {label:<24} {values['n']:>6}  "
                f"{values.get('BLEU-1', 0):.4f}   {values.get('BLEU-4', 0):.4f}   "
                f"{values.get('CIDEr-D', 0):.4f}"
            )

    scorable = [p for p in predictions if p["references"]]
    if examples and scorable:
        print(f"\n  {min(examples, len(scorable))} example captions")
        step = max(1, len(scorable) // examples)
        for record in scorable[::step][:examples]:
            print(f"    PRED: {record['prediction']}")
            print(f"    REF : {record['references'][0]}")
            print()


if __name__ == "__main__":
    raise SystemExit(main())
