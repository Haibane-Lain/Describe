"""Assert the live JPEG path and the cached-feature path produce the same captions.

Two code paths produce captions in this project: ``decode_cached`` reads
precomputed vision features, and :mod:`train.finetune.predict` runs the vision
tower on a real photograph. Evaluation uses the first; the product uses the
second. If they diverge - a different resize, a different normalisation, a
different prompt token, different beam settings - the reported metrics stop
describing the thing that ships.

This is the same class of check that caught the 577-vs-576 image-token bug,
which was invisible until real data hit it.

    py -3.12 -m train.finetune.verify_live --run-dir runs/blip_vizwiz_lora --limit 20
"""

from __future__ import annotations

import sys
import time
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
from train.finetune.cache import (  # noqa: E402
    CachePaths,
    image_path_for,
    index_images,
    iter_manifest,
    load_cache,
)
from train.finetune.dataset import build_split  # noqa: E402
from train.finetune.evaluate_blip import decode_cached  # noqa: E402
from train.finetune.predict import BlipCaptioner  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


def normalise(text: str) -> str:
    return " ".join(text.lower().strip().split())


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Live vs cached caption equivalence.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-tensor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--min-agreement", type=float, default=0.9,
                        help="floor for end-to-end caption agreement; the strict same-feature check "
                             "must be exact, but the JPEG path re-encodes in fp32 against an fp16 "
                             "cache, which can flip a beam")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--bf16", action="store_true",
                        help="compare under bf16 instead of fp32. Captions will NOT match exactly: "
                             "bf16 perturbs the encoder features by ~0.03%% (cosine 0.9997) and beam "
                             "search amplifies that along the sequence, so the two paths pick different "
                             "beams. fp32 is the correct setting for asserting equivalence; bf16 is the "
                             "faster setting for production and for the reported metrics.")
    args = parser.parse_args(argv)

    configure_for_speed()
    info = get_device()
    device = info.device
    print(f"device     : {info.name} ({info.backend})")

    # ---- cached path
    features, index = load_cache(CachePaths(index=args.cache_index, tensor=args.cache_tensor))
    rows = build_split(index)
    if not rows:
        raise SystemExit("no rows with captions in the cache")
    step = max(1, len(rows) // args.limit)
    rows = rows[::step][: args.limit]

    model, report = build_model(resolve_model_id(args.model), with_vision=True, dtype=torch.float32, device=device)

    from train.finetune.base import prepare_model_for_checkpoint

    candidate_paths = [
        p for p in (args.checkpoint, args.run_dir / "best.pt", Path("checkpoints") / args.run_dir.name / "best.pt")
        if p is not None
    ]
    resolved = next((p for p in candidate_paths if Path(p).exists()), None)
    if resolved is None:
        raise SystemExit(f"no checkpoint found for run {args.run_dir}")
    payload_check = torch.load(resolved, map_location="cpu", weights_only=False)

    # Attach the adapter BEFORE loading. `prepare_model_for_checkpoint` only
    # shapes the model (attaches LoRA modules, unfreezes the decoder); it does
    # not load weights. This call was accidentally dropped during an edit, and
    # the result was a silent wrong answer rather than an error: the checkpoint
    # was loaded into a model with no adapter modules, so the load "succeeded"
    # by writing each base weight back onto itself, and the cached path decoded
    # with the unadapted model while reporting success.
    prepare_model_for_checkpoint(model, payload_check)

    loaded, skipped = load_trainable_state_dict(model, payload_check.get("model", payload_check))
    if not loaded:
        raise SystemExit("loaded 0 adapter tensors in the cached path; the comparison would be "
                         "against the unadapted model")
    adapter_keys = [k for k in loaded if "lora_" in k]
    if not adapter_keys:
        raise SystemExit(
            f"loaded {len(loaded)} tensors but none of them are adapter weights. The checkpoint "
            "was almost certainly loaded into a model without LoRA modules attached, which makes "
            "the load a silent no-op that reports success."
        )
    print(f"cached path adapter: {len(loaded)} tensors ({len(adapter_keys)} adapter weights)")
    model.eval()

    from transformers import AutoTokenizer

    model_id = resolve_model_id(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    started = time.time()
    cached_captions = decode_cached(
        model, rows, features, tokenizer, device,
        batch_size=args.limit, num_beams=args.num_beams, max_new_tokens=30, min_new_tokens=2,
        bf16=args.bf16,
    )
    cached_secs = time.time() - started
    print(f"cached first: {cached_captions[0]!r}")
    del model
    torch.cuda.empty_cache()

    # ---- live path, from the checkpoint, on real JPEGs
    captioner = BlipCaptioner.from_run(args.run_dir, model=args.model, checkpoint=args.checkpoint, device=device)

    image_index = index_images(args.image_root)
    manifest_rows = list(iter_manifest(args.manifest))
    by_key = {r.get("file_name") or r.get("image"): r for r in manifest_rows}

    # ---- Check 1: strict equivalence of the decoding logic.
    #
    # Feed the *same* fp16 cached features through the live model and through
    # decode_cached. This isolates the one thing that must be identical - prompt
    # tokens, EOS choice, beam configuration, token trimming - from numerical
    # noise, and any difference here is a real bug.
    same_features = torch.from_numpy(
        np.asarray(features[[int(r["feature_row"]) for r in rows]], dtype=np.float32)
    ).to(device)
    strict = [c.text for c in captioner.caption_batch([], image_embeds=same_features,
                                                      num_beams=args.num_beams, bf16=args.bf16)]
    strict_mismatches = [
        (row.get("file_name"), cached, live)
        for row, cached, live in zip(rows, cached_captions, strict)
        if normalise(cached) != normalise(live)
    ]

    # ---- Check 2: the real end-to-end path, JPEG in.
    #
    # Features here are re-encoded in fp32 rather than read from the fp16 cache,
    # so a small number of captions can legitimately differ: the encoder output
    # moves by ~0.03% and beam search amplifies that into a different beam. That
    # is a documented precision effect, not a wiring error, so this check reports
    # an agreement ratio and a floor rather than demanding equality.
    live_captions: list[str] = []
    timings: list[float] = []
    for row in rows:
        key = row.get("file_name") or row.get("image")
        manifest_row = by_key.get(key, row)
        path = image_path_for(manifest_row, args.manifest, None, args.image_root, image_index)
        result = captioner.caption(path, num_beams=args.num_beams, bf16=args.bf16)
        live_captions.append(result.text)
        timings.append(result.latency_ms)

    mismatches = [
        (row.get("file_name"), cached, live)
        for row, cached, live in zip(rows, cached_captions, live_captions)
        if normalise(cached) != normalise(live)
    ]

    print(f"\ncached path: {len(cached_captions)} captions in {cached_secs:.1f}s "
          f"({1000 * cached_secs / max(len(cached_captions), 1):.0f} ms/img)")
    print(f"live path  : {len(live_captions)} captions, "
          f"{float(np.mean(timings)):.0f} ms/img mean, {float(np.max(timings)):.0f} ms max")
    print(f"\ncheck 1 - same features, both paths : {len(rows) - len(strict_mismatches)}/{len(rows)} identical")
    print(f"check 2 - JPEG end to end           : {len(rows) - len(mismatches)}/{len(rows)} identical "
          f"(fp16 cache vs fp32 re-encode)")

    for key, cached, live in strict_mismatches[:10]:
        print(f"  STRICT MISMATCH {key}")
        print(f"    cached: {cached}")
        print(f"    live  : {live}")

    strict_ok = not strict_mismatches
    agreement = (len(rows) - len(mismatches)) / max(len(rows), 1)
    realistic_ok = agreement >= args.min_agreement

    print(f"\n{'PASS' if strict_ok else 'FAIL'} decoding logic is equivalent")
    print(f"{'PASS' if realistic_ok else 'FAIL'} end-to-end agreement {agreement:.2f} "
          f"(floor {args.min_agreement:.2f})")
    return 0 if (strict_ok and realistic_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
