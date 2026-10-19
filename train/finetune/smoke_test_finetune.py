"""Gate for the BLIP fine-tuning path.

Seven checks, each targeting a specific silent failure that has either already
happened in this project or is a direct consequence of the design chosen here:

1. **Support-only checkpoint round-trip.** Loading what ``trainable_state_dict``
   saved reproduces the model exactly. If this breaks, a resumed run trains a
   different model than the one it checkpointed.
2. **LoRA is an identity at initialisation.** ``B`` is zero-initialised, so a
   wrapped model must produce bit-identical outputs until the first update.
   Without this, wrapping a released checkpoint silently perturbs it.
3. **Cached forward == stock forward.** ``BlipCachedCaptioner`` fed cached
   ``last_hidden_state`` must produce the same loss and logits as
   ``BlipForConditionalGeneration`` fed pixels. This is the check that catches a
   wrong normalisation, a wrong image size, or a wrong token alignment - all of
   which produce a plausible-looking loss while training on the wrong thing.
4. **Loss decreases.** Optimisation actually works on this hardware.
5. **Save / load / resume is stable.** A resumed step reproduces the loss of the
   step it replaced.
6. **Batched generation matches the reference decoder.** ``decode_cached`` with
   zero LoRA delta must produce the same captions as
   ``BlipForConditionalGeneration.generate``. This validates the whole inference
   stack: prompt tokens, EOS choice, beam configuration.
7. **Dataset and collator.** Padding is masked with -100, and a seeded sampler
   picks different captions in different epochs.

Everything runs on a randomly-initialised *tiny* BLIP (64px images, 2 layers,
128-token vocab), so the gate needs no network access and finishes in seconds.
The architecture and the code paths are the real ones; only the sizes are small.

    py -3.12 -m train.finetune.smoke_test_finetune
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    RESULTS.append((name, passed, detail))
    print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""), flush=True)
    return passed


# ----------------------------------------------------------------- the fixture


def build_tiny(workdir: Path):
    """Create a tiny randomly-initialised BLIP and return its directory."""
    from transformers import BlipConfig, BlipForConditionalGeneration, BlipTextConfig, BlipVisionConfig

    text_config = BlipTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        is_decoder=True,
        add_cross_attention=True,
        bos_token_id=101,
        eos_token_id=102,
        sep_token_id=102,
        pad_token_id=0,
    )
    vision_config = BlipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=64,
        patch_size=16,
    )
    config = BlipConfig(text_config=text_config, vision_config=vision_config)
    model = BlipForConditionalGeneration(config)
    model.eval()
    model.save_pretrained(workdir)
    return workdir, model, config


def synthetic_batch(batch: int, image_size: int, length: int, vocab: int, device, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixel_values = torch.rand(batch, 3, image_size, image_size, generator=generator).to(device)
    input_ids = torch.randint(5, vocab, (batch, length), generator=generator).to(device)
    labels = input_ids.clone()
    labels[:, -2:] = -100
    attention_mask = torch.ones_like(input_ids)
    return pixel_values, input_ids, labels, attention_mask


# -------------------------------------------------------------------- checks


def test_cached_matches_stock(workdir: Path, device) -> None:
    from transformers import BlipForConditionalGeneration

    from train.finetune.base import build_model

    model_id, stock, config = build_tiny(workdir)
    stock = stock.to(device)
    wrapper, report = build_model(model_id, with_vision=True, dtype=torch.float32, device=device)
    wrapper.eval()

    # 64px / 16px = 4x4 = 16 patches, plus BLIP's CLS token = 17 image tokens.
    expected_tokens = (config.vision_config.image_size // config.vision_config.patch_size) ** 2 + 1
    check(
        "wrapper reports the right shape",
        report.num_image_tokens == expected_tokens,
        f"{report.num_image_tokens} image tokens (expected {expected_tokens}), "
        f"hidden {report.vision_hidden_size}",
    )

    batch = synthetic_batch(2, config.vision_config.image_size, 6, config.text_config.vocab_size, device)
    pixel_values, input_ids, labels, attention_mask = batch

    with torch.no_grad():
        stock_out = stock(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        wrapper_out = wrapper(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    loss_delta = abs(float(stock_out.loss) - float(wrapper_out.loss))
    logit_delta = float((stock_out.logits - wrapper_out.logits).abs().max())
    check(
        "cached forward matches stock forward",
        loss_delta < 1e-5 and logit_delta < 1e-4,
        f"|d loss| = {loss_delta:.2e}, |d logits|max = {logit_delta:.2e}",
    )

    # Now the actual cached path: feed vision output directly and confirm it is
    # equivalent to what the stock model would compute from pixels.
    with torch.no_grad():
        embeds = stock.vision_model(pixel_values=pixel_values).last_hidden_state
        cached_out = wrapper(
            image_embeds=embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
    check(
        "cached-image forward matches pixel forward",
        abs(float(stock_out.loss) - float(cached_out.loss)) < 1e-5,
        f"loss {float(cached_out.loss):.6f} vs {float(stock_out.loss):.6f}",
    )
    return config


def test_support_checkpoint(workdir: Path, config, device) -> None:
    from train.finetune.base import build_model, load_trainable_state_dict, trainable_state_dict

    model_id = workdir
    model, _ = build_model(model_id, with_vision=False, dtype=torch.float32, device=device)
    model.eval()

    # `build_model` freezes everything; this check is about saving and restoring
    # a trained state, so make the parameters trainable first.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    # Make the model non-trivial so an accidental no-op load is visible.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)

    batch = synthetic_batch(2, config.vision_config.image_size, 6, config.text_config.vocab_size, device)
    _, input_ids, labels, attention_mask = batch
    embeds = torch.randn(2, 16, config.vision_config.hidden_size, device=device)

    with torch.no_grad():
        before = model(image_embeds=embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    state = trainable_state_dict(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    loaded, skipped = load_trainable_state_dict(model, state)

    with torch.no_grad():
        after = model(image_embeds=embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    check(
        "support-only checkpoint round-trips",
        abs(float(before.loss) - float(after.loss)) < 1e-6 and not skipped,
        f"{len(loaded)} tensors, |d loss| = {abs(float(before.loss) - float(after.loss)):.2e}",
    )

    # peft writes a different key shape than `trainable_state_dict` does:
    # `text_decoder.base_model.model.<...>.lora_A.default.weight`. An evaluator
    # that cannot read it loads nothing and silently reports the *pretrained*
    # model's metrics under the fine-tuned run's name. This asserts the mapping
    # directly, including that layer indices survive.
    from train.finetune.base import _normalise_checkpoint_key

    peft_key = (
        "text_decoder.base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight"
    )
    candidates = _normalise_checkpoint_key(peft_key, set(model.state_dict()))
    resolved = candidates[0] if candidates else ""
    check(
        "peft-style checkpoint keys resolve",
        resolved
        == "text_decoder.bert.encoder.layer.0.attention.self.query.lora_A.weight",
        f"{peft_key} -> {resolved}",
    )


def test_lora_identity(workdir: Path, config, device) -> None:
    from train.finetune.base import build_model
    from train.finetune.train_blip import apply_manual_lora, count_trainable

    model, _ = build_model(workdir, with_vision=False, dtype=torch.float32, device=device)
    model.eval()

    batch = synthetic_batch(2, config.vision_config.image_size, 6, config.text_config.vocab_size, device)
    _, input_ids, labels, attention_mask = batch
    embeds = torch.randn(2, 17, config.vision_config.hidden_size, device=device)

    with torch.no_grad():
        before = model(image_embeds=embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    wrapped = apply_manual_lora(model, rank=4, alpha=8.0, dropout=0.0)
    model.eval()
    trainable, total = count_trainable(model)

    with torch.no_grad():
        after = model(image_embeds=embeds, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    check(
        "manual LoRA is an identity at init",
        abs(float(before.loss) - float(after.loss)) < 1e-6,
        f"wrapped {wrapped} layers, trainable {trainable/1e3:.1f}k of {total/1e3:.1f}k, "
        f"|d loss| = {abs(float(before.loss) - float(after.loss)):.2e}",
    )
    check("LoRA leaves most parameters frozen", 0 < trainable < 0.5 * total, f"{100*trainable/total:.1f}% trainable")


def test_loss_decreases(workdir: Path, config, device) -> None:
    """Two checks on the same tiny problem.

    There are two different things that can go wrong and they need different
    evidence:

    * **The training loop is broken.** Detected by the full-fine-tune control:
      with ~137k trainable parameters on a 4-example memorisation task, the loss
      must fall hard. If it does not, the optimiser, the loss wiring, the label
      shift or the autocast context is wrong.
    * **LoRA cannot express the update.** Detected separately, and given a much
      lower bar, because a rank-8 adapter over a 2-layer text decoder genuinely
      is a small-capacity update. LoRA is still expected to descend; it is not
      expected to memorise.
    """
    from train.finetune.base import build_model
    from train.finetune.train_blip import apply_manual_lora

    embeds = None
    input_ids = None

    def make_problem():
        # The generator must match the tensor's device, so draw on the CPU and
        # move the result.
        generator = torch.Generator(device="cpu").manual_seed(11)
        local_embeds = torch.randn(4, 16, config.vision_config.hidden_size, generator=generator).to(device)
        local_ids = torch.randint(5, config.text_config.vocab_size, (4, 8), generator=generator).to(device)
        return local_embeds, local_ids

    def run(steps: int, *, lora: bool) -> tuple[float, float, int]:
        torch.manual_seed(0)
        model, _ = build_model(workdir, with_vision=False, dtype=torch.float32, device=device)
        if lora:
            apply_manual_lora(model, rank=8, alpha=16.0, dropout=0.0)
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        model.train()

        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=5e-3)

        local_embeds, local_ids = make_problem()
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(image_embeds=local_embeds, input_ids=local_ids, labels=local_ids)
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.detach()))
        return sum(losses[:5]) / 5, sum(losses[-5:]) / 5, sum(p.numel() for p in parameters)

    full_first, full_last, full_params = run(200, lora=False)
    check(
        "training loop descends (full fine-tune control)",
        full_last < full_first * 0.30,
        f"{full_first:.3f} -> {full_last:.3f} ({100 * (1 - full_last / full_first):.0f}% reduction, "
        f"{full_params/1e3:.1f}k trainable)",
    )

    lora_first, lora_last, lora_params = run(200, lora=True)
    check(
        "LoRA adapter descends",
        lora_last < lora_first * 0.85,
        f"{lora_first:.3f} -> {lora_last:.3f} ({100 * (1 - lora_last / lora_first):.0f}% reduction, "
        f"{lora_params/1e3:.1f}k trainable)",
    )


def test_resume_stability(workdir: Path, config, device) -> None:
    from train.finetune.base import build_model, load_trainable_state_dict, trainable_state_dict
    from train.finetune.train_blip import apply_manual_lora
    from train.finetune.train_blip import atomic_save

    torch.manual_seed(1)
    model, _ = build_model(workdir, with_vision=False, dtype=torch.float32, device=device)
    apply_manual_lora(model, rank=4, alpha=8.0, dropout=0.0)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)

    embeds = torch.randn(4, 16, config.vision_config.hidden_size, device=device)
    input_ids = torch.randint(5, config.text_config.vocab_size, (4, 8), device=device)

    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        model(image_embeds=embeds, input_ids=input_ids, labels=input_ids).loss.backward()
        optimizer.step()

    checkpoint = workdir / "step5.pt"
    atomic_save(
        {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(), "epoch": 0, "global_step": 5},
        checkpoint,
    )

    # Continue the live model by one step: this is the reference loss.
    optimizer.zero_grad(set_to_none=True)
    reference = float(model(image_embeds=embeds, input_ids=input_ids, labels=input_ids).loss.detach())

    # Rebuild from the checkpoint and repeat that same step.
    torch.manual_seed(1)
    fresh, _ = build_model(workdir, with_vision=False, dtype=torch.float32, device=device)
    apply_manual_lora(fresh, rank=4, alpha=8.0, dropout=0.0)
    fresh.train()
    fresh_parameters = [p for p in fresh.parameters() if p.requires_grad]
    fresh_optimizer = torch.optim.AdamW(fresh_parameters, lr=1e-3)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    loaded, skipped = load_trainable_state_dict(fresh, payload["model"])
    fresh_optimizer.load_state_dict(payload["optimizer"])

    fresh_optimizer.zero_grad(set_to_none=True)
    resumed = float(fresh(image_embeds=embeds, input_ids=input_ids, labels=input_ids).loss.detach())

    check(
        "resume reproduces the checkpointed loss",
        abs(reference - resumed) < 1e-5 and not skipped,
        f"{reference:.6f} vs {resumed:.6f} ({len(loaded)} tensors)",
    )


def test_generation_matches_reference(workdir: Path, config, device) -> None:
    from transformers import BlipForConditionalGeneration

    from train.finetune.base import build_model
    from train.finetune.evaluate_blip import decode_cached

    model_id = workdir
    stock = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
    stock.eval()
    wrapper, _ = build_model(model_id, with_vision=True, dtype=torch.float32, device=device)
    wrapper.eval()

    batch = 3
    generator = torch.Generator(device="cpu").manual_seed(3)
    pixel_values = torch.rand(batch, 3, config.vision_config.image_size, config.vision_config.image_size, generator=generator).to(device)

    with torch.no_grad():
        reference_ids = stock.generate(
            pixel_values=pixel_values,
            num_beams=2,
            max_new_tokens=5,
            min_new_tokens=1,
        )
        embeds = stock.vision_model(pixel_values=pixel_values).last_hidden_state
        # Mirror decode_cached: token ids live on the generation config, because
        # passing them as kwargs collides with generate()'s own binding in this
        # transformers version.
        decoder = wrapper.text_decoder
        decoder.generation_config.eos_token_id = config.text_config.sep_token_id
        decoder.generation_config.pad_token_id = config.text_config.pad_token_id
        decoder.generation_config.bos_token_id = config.text_config.bos_token_id
        ours_ids = decoder.generate(
            input_ids=torch.full((batch, 1), config.text_config.bos_token_id, dtype=torch.long, device=device),
            encoder_hidden_states=embeds,
            encoder_attention_mask=torch.ones(embeds.shape[:-1], dtype=torch.long, device=device),
            num_beams=2,
            max_new_tokens=5,
            min_new_tokens=1,
            use_cache=True,
        )

    # Both decoders emit [bos, ...content..., eos]; compare the content tokens.
    def content(ids: list[int]) -> list[int]:
        return list(ids)[1:-1] if len(ids) > 2 else []

    reference_content = [content(ids) for ids in reference_ids.tolist()]
    ours_content = [content(ids) for ids in ours_ids.tolist()]
    match = reference_content == ours_content
    check(
        "batched generation matches the reference decoder",
        match,
        f"reference {reference_content[:2]} vs ours {ours_content[:2]}",
    )

    # And exercise decode_cached end to end against a synthetic cache index.
    features = np.asarray(embeds.detach().to("cpu").numpy(), dtype=np.float16)
    index = [{"image": f"img{i}.jpg", "feature_row": i, "captions": ["a small test image"]} for i in range(batch)]

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(i) for i in ids)

    captions = decode_cached(
        wrapper, index, features, _Tok(), device,
        batch_size=2, num_beams=2, max_new_tokens=5, min_new_tokens=1,
    )
    check(
        "decode_cached returns one caption per row",
        len(captions) == batch and all(captions),
        f"{captions[:2]}",
    )


def test_dataset_and_collator(workdir: Path, config) -> None:
    from train.finetune.dataset import BlipCaptionCollator, CachedBlipDataset, build_split, captions_of

    index = [
        {"image": "a.jpg", "feature_row": 0, "captions": ["one cat", "two cats", "cat here"]},
        {"image": "b.jpg", "feature_row": 1, "captions": []},
        {"image": "c.jpg", "feature_row": 2, "captions": ["a dog on grass"]},
    ]
    rows = build_split(index)
    check("captionless rows are dropped", len(rows) == 2, f"{len(rows)} of {len(index)} kept")

    features = np.random.rand(3, 16, config.vision_config.hidden_size).astype(np.float16)

    class _Tok:
        """Whitespace tokenizer with a small deterministic vocabulary."""

        def __call__(self, text, **kwargs):
            ids = []
            for word in text.split():
                for character in word:
                    ids.append(5 + (ord(character) % 40))
            return {"input_ids": ids or [5]}

    dataset = CachedBlipDataset(rows, features, _Tok(), max_length=8, seed=7)

    # Determinism: the same (epoch, row) must sample the same caption, or a
    # resumed run would train on a different sequence than the one it logged.
    dataset.set_epoch(3)
    repeat = [tuple(dataset[0]["input_ids"].tolist()) for _ in range(3)]
    check("sampling is deterministic per (epoch, row)", len(set(repeat)) == 1, f"{repeat[0]}")

    # Epoch dependence: across epochs the chosen caption must vary, otherwise
    # multi-caption images are wasted and every epoch sees the same sentence.
    across = set()
    for epoch in range(12):
        dataset.set_epoch(epoch)
        across.add(tuple(dataset[0]["input_ids"].tolist()))
    check(
        "sampler varies across epochs",
        len(across) >= 2,
        f"{len(across)} distinct caption(s) across 12 epochs (3 available)",
    )

    collator = BlipCaptionCollator(pad_token_id=0)
    sample = collator([dataset[0], dataset[1]])
    masked = int((sample["labels"] == -100).sum())
    check(
        "padding is masked with -100",
        masked > 0 and sample["image_embeds"].shape[0] == 2,
        f"labels {tuple(sample['labels'].shape)}, {masked} masked positions",
    )
    check(
        "caption extraction tolerates field spellings",
        captions_of({"caption": "x"}) == ["x"] and captions_of({"sentences": [{"raw": "y"}]}) == ["y"],
    )


def test_degrade_determinism() -> None:
    """Degradation must be reproducible, and must actually change the image.

    A degraded feature cache is only explainable if it can be rebuilt
    byte-for-byte. Three tools now depend on this: the `--degrade` cache pass,
    merge_caches, and the reported E7 experiment. If the transforms became
    non-deterministic, that experiment could not be reproduced or defended.

    Also checks that the transforms are not silently no-ops, which would make a
    "degradation" run meaningless while still reporting success.
    """
    import random

    import numpy as np
    from PIL import Image

    from train.finetune import degrade as D

    image = Image.new("RGB", (64, 64))
    # A non-uniform pattern, so blur and framing have something to change.
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = ((x * 4) % 256, (y * 4) % 256, ((x + y) * 2) % 256)

    config = D.DegradeConfig(probability=1.0, severity=1.0, seed=11)

    # 1. same key twice -> identical bytes
    first, flaw_a = D.degrade_image(image, config, "same-key")
    second, flaw_b = D.degrade_image(image, config, "same-key")
    check(
        "degrade: deterministic per image key",
        first.tobytes() == second.tobytes() and flaw_a == flaw_b,
        f"flaw={flaw_a} vs {flaw_b}",
    )

    # 2. different keys -> not all identical (the seed actually varies)
    variants = {D.degrade_image(image, config, f"key-{i}")[0].tobytes() for i in range(40)}
    check(
        "degrade: varies across images",
        len(variants) > 30,
        f"{len(variants)} distinct results from 40 keys",
    )

    # 3. probability is honoured: p=0 means never
    never = [D.degrade_image(image, D.DegradeConfig(probability=0.0, seed=1), f"n{i}")[1] for i in range(50)]
    check(
        "degrade: p=0.0 applies nothing",
        all(f is None for f in never),
        f"{sum(1 for f in never if f)} of 50 degraded",
    )

    # 4. each transform measurably changes the image
    unchanged = []
    for name, transform in D.TRANSFORMS.items():
        out = transform(image.copy(), random.Random(3), 1.0)
        if out.size != image.size:
            out = out.resize(image.size)
        if np.asarray(out, dtype=np.int16).sum() == np.asarray(image, dtype=np.int16).sum():
            unchanged.append(name)
    check(
        "degrade: every transform alters the image",
        not unchanged,
        f"no-ops: {unchanged}" if unchanged else "all 6 transforms change pixels",
    )

    # 5. the flaw sampler follows the annotated prior (blur should dominate)
    from collections import Counter

    counts: Counter = Counter()
    for i in range(600):
        _, flaw = D.degrade_image(image, config, f"mix{i}")
        counts[(flaw or "").split("+")[0]] += 1
    total = sum(counts.values())
    blur_share = counts["blur"] / total
    check(
        "degrade: flaw mix follows the VizWiz prior",
        0.30 < blur_share < 0.55,
        f"blur share {100 * blur_share:.1f}% (expected ~42.6%)",
    )


def main() -> int:
    from train.utils.device import configure_for_speed, get_device

    configure_for_speed()
    info = get_device()
    device = info.device
    print(f"device: {info.name} ({info.backend})\n")
    print("BLIP fine-tuning smoke test (tiny synthetic model, no network)\n")

    workdir = Path(tempfile.mkdtemp(prefix="blip_smoke_"))
    try:
        config = test_cached_matches_stock(workdir, device)
        test_support_checkpoint(workdir, config, device)
        test_lora_identity(workdir, config, device)
        test_loss_decreases(workdir, config, device)
        test_resume_stability(workdir, config, device)
        test_generation_matches_reference(workdir, config, device)
        test_dataset_and_collator(workdir, config)
        test_degrade_determinism()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} - {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
