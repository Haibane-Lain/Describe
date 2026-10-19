"""Fine-tune BLIP on cached image features.

Run this after ``py -3.12 -m train.finetune.cache`` has written a feature cache.
Training then never reads an image: the vision tower's output is replayed from
disk, so an epoch is pure text forward/backward.

What is trained is an explicit choice, and the default is the cheapest one that
works:

``--tune lora``  (default)
    LoRA on the text decoder's attention and feed-forward projections. A few
    million trainable parameters, tiny optimizer state, negligible risk of
    destroying the pretrained language model. Requires ``peft``.

``--tune lora-manual``
    The same thing with a ~40-line local implementation
    (:class:`ManualLoRA`). Use this if ``peft`` fights this transformers
    version; the training loop is otherwise identical.

``--tune head``
    The entire text decoder and cross-attention are trained, vision frozen.
    ~92 M parameters for BLIP-base. Use when LoRA plateaus below target.

None of these unfreeze the vision tower. Training it requires the raw images,
loses the feature cache, and costs roughly 3x per step - and this project
already has direct evidence that unfreezing a pretrained encoder is not
automatically an improvement (``flickr30k_e2e_safe`` scored *worse* than the
frozen-encoder model it replaced).

Stability notes, all of which are documented failure modes on this machine
rather than theoretical concerns:

* Checkpoints are written every ``--save-every`` steps, atomically. A HIP
  driver fault (``0xC0000005`` in ``amdhip64_7.dll``) has killed a completed
  epoch's work here before; ``--resume`` picks up from the last step.
* Batch sizes come from ``train.bench_blip``, not from optimism. Use
  ``--grad-accum`` to reach an effective batch the GPU cannot hold at once.
* ``configure_for_speed()`` disables MIOpen on RDNA4, which is required because
  the Windows ROCm SDK is missing the rocrand headers its dropout kernel needs.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.base import (  # noqa: E402
    DEFAULT_LORA_LEAVES,
    build_model,
    load_trainable_state_dict,
    resolve_model_id,
    text_decoder_lora_targets,
    trainable_state_dict,
)
from train.finetune.cache import CachePaths, load_cache, row_image_key  # noqa: E402
from train.finetune.dataset import (  # noqa: E402
    BlipCaptionCollator,
    CachedBlipDataset,
    build_split,
    describe_split,
    load_index,
)
from train.utils.device import configure_for_speed, get_device, set_seed  # noqa: E402

# --------------------------------------------------------------------- LoRA


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper: ``y = W x + (alpha / r) B A x``.

    ``B`` is zero-initialised, so a freshly wrapped model is bit-identical to the
    unwrapped one. That property is asserted in ``smoke_test_finetune.py`` and is
    what makes it safe to wrap a released checkpoint in place.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # New parameters are created on the CPU. If the wrapped layer already
        # lives on the GPU, they must follow it or the first forward pass dies
        # with "mat2 is on cpu" - an easy mistake, and the reason the smoke test
        # exercises LoRA on the actual device.
        device = base.weight.device

        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=base.weight.dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling


def apply_manual_lora(
    model,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    leaves: set[str] | None = None,
) -> int:
    """Wrap the text decoder's linear layers with :class:`LoRALinear`.

    ``leaves`` holds leaf module names, matching
    :func:`train.finetune.base.text_decoder_lora_targets`. Only modules inside
    ``model.text_decoder`` are touched, so the vision tower (if present) is
    never wrapped.

    Returns the number of layers wrapped.
    """
    leaves = leaves if leaves is not None else text_decoder_lora_targets()
    wrapped = 0
    for module_name, module in model.text_decoder.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in leaves and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                wrapped += 1
    return wrapped


def apply_peft_lora(model, *, rank: int, alpha: float, dropout: float, leaves: set[str] | None = None):
    """Wrap the text decoder with ``peft``, leaving the outer module untouched.

    Wrapping the *whole* ``BlipCachedCaptioner`` does not work: ``peft``'s
    tuner installs its own ``forward`` whose signature takes ``input_ids``,
    ``attention_mask``, ``inputs_embeds`` and ``**kwargs``. Our two extra
    arguments - ``image_embeds`` and ``labels`` - are swallowed by that
    ``**kwargs`` and never reach the wrapper, which then fails with
    "forward() got an unexpected keyword argument". Wrapping only
    ``text_decoder`` keeps the outer call signature ours.

    Returns the adapted model, or ``None`` if ``peft`` is unavailable.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        return None

    config = LoraConfig(
        r=rank,
        lora_alpha=int(alpha),
        lora_dropout=dropout,
        bias="none",
        # Leaf names, matching the built-in implementation in this module.
        # Attention projections by default; see text_decoder_lora_targets.
        target_modules=sorted(leaves if leaves is not None else DEFAULT_LORA_LEAVES),
        task_type="CAUSAL_LM",
    )
    model.text_decoder = get_peft_model(model.text_decoder, config)
    return model


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ------------------------------------------------------------------ training


@dataclass
class RunState:
    epoch: int = 0
    global_step: int = 0
    best_score: float = -1.0
    history: list[dict] = field(default_factory=list)


def cosine_with_warmup(step: int, *, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = min(1.0, (step - warmup) / (total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def bleu_blend(metrics: dict) -> float:
    """Checkpoint-selection score.

    Why not BLEU-4 alone: it is exactly zero for most of a run, so the first
    evaluated epoch would win forever (this bug is recorded in
    ``train/docs/training-status.md``). Why not loss alone: captioning
    cross-entropy plateaus well before quality does. The blend stays
    discriminative while BLEU-4 is still zero and converges to BLEU-4 later.
    """
    return (
        0.15 * metrics.get("BLEU-1", 0.0)
        + 0.25 * metrics.get("BLEU-2", 0.0)
        + 0.25 * metrics.get("BLEU-3", 0.0)
        + 0.35 * metrics.get("BLEU-4", 0.0)
    )


def atomic_save(payload: dict, path: Path) -> None:
    """Write to a temp file and replace, so a crash mid-save cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def run_validation(model, rows, features, tokenizer, device, args, epoch: int) -> dict:
    from train.finetune.evaluate_blip import decode_cached, evaluate_predictions
    from train.finetune.dataset import captions_of

    captions = decode_cached(
        model, rows, features, tokenizer, device,
        batch_size=args.eval_batch_size,
        num_beams=args.eval_beams,
        max_new_tokens=args.max_new_tokens,
        bf16=not args.no_eval_bf16,
    )
    predictions = [
        {
            "image_id": row.get("image_id") or row.get("id"),
            # Same resolver the cache uses; manifests identify images by
            # `file_name`, and a wrong key here makes the quality/flaw split in
            # diagnose_predictions match nothing while still printing a
            # plausible aggregate.
            "file_name": row_image_key(row) or row.get("image") or row.get("file"),
            "text_detected": row.get("text_detected"),
            "prediction": caption,
            "references": captions_of(row),
        }
        for row, caption in zip(rows[: len(captions)], captions)
    ]
    result = evaluate_predictions(predictions, with_meteor=False, with_spice=False)
    result["epoch"] = epoch
    return result


def train(args) -> int:
    configure_for_speed()
    info = get_device()
    device = torch.device("cpu") if args.cpu else info.device
    set_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_dir = Path("runs") / args.run_name
    ckpt_dir = Path("checkpoints") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(args.model)
    print(f"device     : {info.name} ({info.backend})" if not args.cpu else "device     : cpu")
    print(f"model      : {model_id}")
    print(f"tune       : {args.tune}")

    # ---------------------------------------------------------------- cache
    cache_paths = CachePaths(index=args.cache_index, tensor=args.cache_tensor)
    features, cache_index = load_cache(cache_paths)
    print(f"cache      : {cache_paths.tensor}  shape={tuple(features.shape)}")

    train_rows = build_split(
        cache_index, keep=args.limit_train, sample=args.sample_train, text_detected=args.text_filter
    )
    if args.text_filter is not None:
        print(f"  (restricted to text_detected={args.text_filter} rows)")
    if args.sample_train and args.limit_train:
        print(f"  (train rows sampled at random with seed {args.seed}, not the leading rows)")
    print("  " + describe_split("train", train_rows))

    val_rows: list[dict] = []
    val_features = None
    if args.val_cache_index and args.val_cache_tensor:
        val_features, val_index = load_cache(CachePaths(index=args.val_cache_index, tensor=args.val_cache_tensor))
        # Validation is NOT filtered. Per-epoch selection must reflect the split
        # the model will be judged on, or a text-restricted model would be
        # selected on a text-only subset and could regress everywhere else
        # without it showing up until the final evaluation.
        val_rows = build_split(val_index, keep=args.limit_val, sample=args.sample_val)
        print("  " + describe_split("val", val_rows))
        if args.sample_val and args.limit_val:
            print(f"  (val rows sampled at random with seed {args.seed}; used for per-epoch "
                  f"checkpoint selection only)")

    if not train_rows:
        raise SystemExit("no trainable rows in the cache (every row has zero captions?)")

    # ---------------------------------------------------------------- model
    model, report = build_model(
        model_id,
        with_vision=False,
        fp32_layer_norm=not args.no_fp32_layer_norm,
        dtype=torch.float32,
        device=device,
    )
    if report.num_image_tokens != features.shape[1]:
        raise SystemExit(
            f"cache has {features.shape[1]} image tokens but {model_id} expects "
            f"{report.num_image_tokens}; rebuild the cache with --model {args.model}"
        )
    print(f"text params: {report.num_parameters_text_decoder/1e6:.1f} M")

    if args.tune in ("lora", "lora-manual"):
        leaves = set(args.lora_leaves) if args.lora_leaves else text_decoder_lora_targets()
        if args.tune == "lora":
            adapted = apply_peft_lora(
                model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout, leaves=leaves
            )
            if adapted is None:
                print("  ! peft is not installed; falling back to the built-in LoRA implementation")
                n = apply_manual_lora(
                    model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout, leaves=leaves
                )
                print(f"  wrapped {n} text-decoder linear layers (manual LoRA) on {sorted(leaves)}")
            else:
                model = adapted
                print(f"  peft LoRA r={args.lora_rank} alpha={args.lora_alpha} on {sorted(leaves)}")
        else:
            n = apply_manual_lora(
                model, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout, leaves=leaves
            )
            print(f"  wrapped {n} text-decoder linear layers (manual LoRA) on {sorted(leaves)}")
    elif args.tune == "head":
        for parameter in model.text_decoder.parameters():
            parameter.requires_grad_(True)
    else:
        raise SystemExit(f"unknown --tune {args.tune!r}; use lora, lora-manual, or head")

    trainable, total = count_trainable(model)
    print(f"trainable  : {trainable/1e6:.2f} M of {total/1e6:.2f} M ({100*trainable/total:.2f}%)")
    if trainable == 0:
        raise SystemExit("no trainable parameters - check --tune and the LoRA target names")

    if args.grad_checkpointing:
        model.text_decoder.gradient_checkpointing_enable()
        if hasattr(model.text_decoder, "enable_input_require_grads"):
            model.text_decoder.enable_input_require_grads()
        print("grad ckpt  : on (text decoder)")

    # ------------------------------------------------------------------ data
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = int(model.config.text_config.pad_token_id)

    train_dataset = CachedBlipDataset(train_rows, features, tokenizer, max_length=args.max_caption_len, seed=args.seed)
    collator = BlipCaptionCollator(pad_id)

    # ------------------------------------------------------------ optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8
    )

    steps_per_epoch = max(1, math.ceil(len(train_rows) / (args.batch_size * args.grad_accum)))
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup if args.warmup is not None else max(10, int(0.03 * total_steps))

    state = RunState()
    checkpoint_path = ckpt_dir / "last.pt"

    if args.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded, skipped = load_trainable_state_dict(model, payload["model"])
        state.epoch = int(payload.get("epoch", 0))
        state.global_step = int(payload.get("global_step", 0))
        state.best_score = float(payload.get("best_score", -1.0))
        state.history = list(payload.get("history", []))
        if "optimizer" in payload:
            try:
                optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, KeyError) as exc:
                print(f"  ! optimizer state not restored ({exc}); continuing with a fresh optimiser")
        print(f"resumed    : epoch {state.epoch}, step {state.global_step}, best {state.best_score:.4f}")
        print(f"             {len(loaded)} tensors loaded, {len(skipped)} skipped")
    elif args.init_from and Path(args.init_from).exists():
        payload = torch.load(Path(args.init_from), map_location="cpu", weights_only=False)
        loaded, skipped = load_trainable_state_dict(model, payload.get("model", payload))
        print(f"init-from  : {args.init_from} ({len(loaded)} tensors, {len(skipped)} skipped)")

    history_path = run_dir / "history.jsonl"
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "tune": args.tune,
                "lora_rank": args.lora_rank if args.tune.startswith("lora") else None,
                "lora_alpha": args.lora_alpha if args.tune.startswith("lora") else None,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "effective_batch": args.batch_size * args.grad_accum,
                "max_caption_len": args.max_caption_len,
                "grad_checkpointing": args.grad_checkpointing,
                "cache": str(cache_paths.tensor),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "device": info.name if not args.cpu else "cpu",
                "backend": info.backend if not args.cpu else "cpu",
                "torch": torch.__version__,
                **report.as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    if args.bf16:
        print("precision  : bf16 autocast")

    print(
        f"\n  steps/epoch {steps_per_epoch}  total steps {total_steps}  "
        f"warmup {warmup}  lr {args.lr}  eff. batch {args.batch_size * args.grad_accum}\n"
    )

    global_step = state.global_step
    for epoch in range(state.epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)

        order = list(range(len(train_dataset)))
        generator = random.Random(args.seed + epoch)
        generator.shuffle(order)

        for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
            items = [train_dataset[i] for i in order[start : start + args.batch_size]]
            if not items:
                continue
            batch = collator(items)
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.autocast("cuda", dtype=dtype, enabled=args.bf16 and device.type == "cuda"):
                output = model(**batch)
                loss = output.loss / args.grad_accum

            loss.backward()
            batch_tokens = int((batch["labels"] != -100).sum().item())
            epoch_loss += float(loss.detach().item()) * args.grad_accum * batch_tokens
            epoch_tokens += batch_tokens

            if (batch_index + 1) % args.grad_accum == 0 or start + args.batch_size >= len(order):
                lr = cosine_with_warmup(global_step, warmup=warmup, total=total_steps, base_lr=args.lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.save_every and global_step % args.save_every == 0:
                    atomic_save(
                        {
                            "model": trainable_state_dict(model),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "global_step": global_step,
                            "best_score": state.best_score,
                            "history": state.history,
                            "config": {"model_id": model_id, "tune": args.tune, "lora_rank": args.lora_rank},
                        },
                        ckpt_dir / "last.pt",
                    )
                    print(f"    checkpoint at step {global_step}", flush=True)

        elapsed = time.time() - started
        train_loss = epoch_loss / max(epoch_tokens, 1)
        print(
            f"  epoch {epoch + 1:>3}/{args.epochs} done | train {train_loss:.4f} | "
            f"{elapsed:6.1f}s | {epoch_tokens / max(elapsed, 1e-9):7.1f} tok/s",
            flush=True,
        )

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "seconds": round(elapsed, 1),
            "global_step": global_step,
        }

        if val_rows and val_features is not None and (epoch + 1) % args.eval_every == 0:
            was_training = model.training
            metrics = run_validation(model, val_rows, val_features, tokenizer, device, args, epoch + 1)
            model.train(was_training)

            core = metrics.get("metrics", {})
            score = bleu_blend(core)
            record.update(
                {
                    "val_bleu1": core.get("BLEU-1"),
                    "val_bleu4": core.get("BLEU-4"),
                    "val_cider": core.get("CIDEr-D"),
                    "val_rouge_l": core.get("ROUGE-L"),
                    "val_score": score,
                    "val_distinct_tokens": metrics.get("diagnostics", {}).get("distinct_tokens"),
                    "val_fourgram_match": metrics.get("diagnostics", {}).get("fourgram_match_rate"),
                }
            )
            print(
                f"    val BLEU-1 {core.get('BLEU-1', 0):.4f}  BLEU-4 {core.get('BLEU-4', 0):.4f}  "
                f"CIDEr-D {core.get('CIDEr-D', 0):.4f}  ROUGE-L {core.get('ROUGE-L', 0):.4f}  "
                f"distinct {metrics.get('diagnostics', {}).get('distinct_tokens')}  "
                f"4-gram {100 * metrics.get('diagnostics', {}).get('fourgram_match_rate', 0):.1f}%",
                flush=True,
            )
            (run_dir / f"val_epoch{epoch + 1}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            if score > state.best_score:
                state.best_score = score
                atomic_save(
                    {
                        "model": trainable_state_dict(model),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_score": score,
                        "metrics": core,
                        "config": {"model_id": model_id, "tune": args.tune, "lora_rank": args.lora_rank},
                    },
                    ckpt_dir / "best.pt",
                )
                print(f"    new best ({score:.4f}) -> {ckpt_dir / 'best.pt'}", flush=True)

        state.history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        atomic_save(
            {
                "model": trainable_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_score": state.best_score,
                "history": state.history,
                "config": {"model_id": model_id, "tune": args.tune, "lora_rank": args.lora_rank},
            },
            ckpt_dir / "last.pt",
        )

    print(f"\ndone: {args.epochs} epochs, best selection score {state.best_score:.4f}")
    print(f"  checkpoints: {ckpt_dir}")
    print(f"  history    : {history_path}")
    return 0


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune BLIP on cached image features.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="base", help="'base', 'large', or a Hub repo id")
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--cache-tensor", type=Path, required=True)
    parser.add_argument("--val-cache-index", type=Path, default=None)
    parser.add_argument("--val-cache-tensor", type=Path, default=None)

    parser.add_argument("--tune", default="lora", choices=["lora", "lora-manual", "head"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-leaves", nargs="+", default=None,
                        help="leaf module names to adapt; default is query key value. Adding 'dense' "
                             "also adapts the attention output and both MLP projections, which roughly "
                             "triples the trainable parameters and sits between narrow LoRA and a full "
                             "decoder fine-tune")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-beams", type=int, default=5,
                        help="beam width for per-epoch selection; beam-1 (~150 img/s) is the "
                             "practical choice, beam-5 (~9 img/s) is for the final reported number")
    parser.add_argument("--no-eval-bf16", action="store_true",
                        help="disable bf16 autocast during validation generation (about 4x slower)")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-caption-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=None, help="steps; default 3%% of total")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500, help="steps; 0 disables mid-epoch saves")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast (recommended when it fits)")
    parser.add_argument("--no-fp32-layer-norm", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--sample-train", action="store_true",
                        help="draw --limit-train rows at random instead of taking the leading rows")
    parser.add_argument("--sample-val", action="store_true",
                        help="draw --limit-val rows at random; use a fixed subset for per-epoch "
                             "selection and evaluate the full split once at the end")
    parser.add_argument("--text-filter", choices=["true", "false"], default=None,
                        help="train only on rows where text_detected matches. 'true' builds a "
                             "text-aware variant on the 16,272 captioned text-bearing training "
                             "images (E4); validation is deliberately left unfiltered")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-from", default=None, help="checkpoint to initialise from (adapter transfer)")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # `--text-filter` arrives as a string from the CLI; the manifest flag is a
    # real bool, and `is True` comparisons downstream would silently match
    # nothing if a string slipped through.
    if args.text_filter is not None:
        args.text_filter = args.text_filter == "true"
        assert isinstance(args.text_filter, bool), "text_filter must be a bool for `is` matching"
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
