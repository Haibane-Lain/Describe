"""Live JPEG -> caption inference.

Everything else in this package is verified against *precomputed* features. This
module exists because the product does not have precomputed features: the
backend receives an uploaded photograph and has to produce a caption from it.
That is a different code path, and a silently divergent one is exactly the kind
of bug that surfaces during a demo.

So this module deliberately reuses the same pieces as the cached path - the same
``_preprocess`` normalisation, the same beam settings, the same token contract -
and ``verify_live.py`` asserts the two paths agree caption-for-caption on real
images. If they ever drift, that check fails.

    from train.finetune.predict import BlipCaptioner
    captioner = BlipCaptioner.from_run("runs/blip_vizwiz_lora", model="base")
    captioner.caption("photo.jpg")           # or a PIL.Image, or raw bytes
    captioner.caption_batch(paths)           # batched

The returned ``Caption`` carries per-image latency, because proposal 5C requires
the latency budget to be reported per stage rather than asserted as one figure.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from train.finetune.base import (
    build_model,
    load_trainable_state_dict,
    prepare_model_for_checkpoint,
    resolve_model_id,
)
from train.finetune.cache import _preprocess
from train.utils.device import configure_for_speed, get_device

DEFAULT_RUN_CONFIG = "run_config.json"


@dataclass
class Caption:
    text: str
    latency_ms: float
    image_embeds_ms: float


def _load_any_image(source) -> "object":
    """Accept a path, raw bytes, or an already-open PIL image."""
    from PIL import Image

    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as handle:
            return handle.convert("RGB").copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def find_checkpoint(run_dir: Path, explicit: Path | None = None) -> Path:
    """Locate a run's checkpoint under either the run or checkpoint root."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"checkpoint not found: {explicit}")
        return explicit
    candidates = [
        run_dir / "best.pt",
        run_dir / "last.pt",
        Path("checkpoints") / run_dir.name / "best.pt",
        Path("checkpoints") / run_dir.name / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(f"    {c}" for c in candidates)
    raise FileNotFoundError(f"no checkpoint for run {run_dir.name}. Tried:\n{tried}")


class BlipCaptioner:
    """A loaded, ready-to-caption BLIP with its adapter applied.

    Construction is expensive (both towers plus a checkpoint read); captioning
    afterwards is not. Build one per process and reuse it.
    """

    def __init__(self, model, tokenizer, device, *, model_id: str, tune: str, checkpoint: Path | None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = model_id
        self.tune = tune
        self.checkpoint = checkpoint
        self.model.eval()

    # ------------------------------------------------------------- construction

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        *,
        model: str = "base",
        checkpoint: str | Path | None = None,
        device=None,
        fp32_layer_norm: bool = True,
    ) -> "BlipCaptioner":
        """Build a captioner from a training run directory.

        The tune mode is read from the run's ``run_config.json`` so callers do
        not have to remember whether a run was LoRA or a full decoder. It is
        read from the *checkpoint payload* as well, and the two must agree -
        guessing here means silently captioning with the pretrained model.
        """
        configure_for_speed()
        run_dir = Path(run_dir)

        if device is None:
            device = get_device().device

        model_name = model
        run_config_path = run_dir / DEFAULT_RUN_CONFIG
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            model_name = run_config.get("model_id") or model_name
        model_name = resolve_model_id(model_name)

        resolved_ckpt = find_checkpoint(run_dir, Path(checkpoint) if checkpoint else None)
        payload = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)

        built, _report = build_model(
            model_name, with_vision=True, fp32_layer_norm=fp32_layer_norm, dtype=torch.float32, device=device
        )
        tune = prepare_model_for_checkpoint(built, payload)

        loaded, skipped = load_trainable_state_dict(built, payload.get("model", payload))
        if not loaded:
            raise SystemExit(
                f"loaded 0 of {len(payload.get('model', {}))} tensors from {resolved_ckpt}; "
                "refusing to caption with the unadapted model while claiming to use the fine-tune"
            )
        print(f"checkpoint : {resolved_ckpt} ({len(loaded)} tensors loaded)")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        built.eval()
        return cls(built, tokenizer, device, model_id=model_name, tune=tune, checkpoint=resolved_ckpt)

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def _encode(self, images: list) -> tuple[torch.Tensor, float]:
        started = time.time()
        pixel_values = _preprocess(images, self.model.config.vision_config.image_size).to(self.device)
        embeds = self.model.encode_image(pixel_values)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return embeds, (time.time() - started) * 1000

    def caption(
        self,
        image,
        *,
        num_beams: int = 5,
        max_new_tokens: int = 30,
        min_new_tokens: int = 2,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        bf16: bool = True,
    ) -> Caption:
        """Caption one photograph."""
        return self.caption_batch(
            [image],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bf16=bf16,
        )[0]

    @torch.no_grad()
    def caption_batch(
        self,
        images: list,
        *,
        num_beams: int = 5,
        max_new_tokens: int = 30,
        min_new_tokens: int = 2,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.15,
        no_repeat_ngram_size: int = 3,
        bf16: bool = True,
        image_embeds: torch.Tensor | None = None,
    ) -> list[Caption]:
        """Caption several photographs, batched.

        Beam settings default to the same values ``decode_cached`` uses. They
        must stay in sync: ``verify_live.py`` compares the two paths, and a
        mismatch here would show up as a quality difference that looks like a
        model problem.

        ``bf16`` defaults to on because it is ~4x faster at beam-5 (see
        ``decode_cached``). Note that bf16 makes captions *not* bit-reproducible:
        it perturbs the encoder output by about 0.03% and beam search amplifies
        that into a different beam choice. That is a precision trade, not a bug,
        but it means exact-equivalence testing must run in fp32.

        ``image_embeds`` bypasses the vision tower. It exists so a verification
        harness can feed *identical* features to both this path and
        ``decode_cached``, isolating the decoding logic from the ~0.03%
        difference between re-encoding an image and reading the fp16 feature
        cache.
        """
        if image_embeds is not None:
            embeds = image_embeds
            embeds_ms = 0.0
            n_images = embeds.shape[0]
        else:
            if not images:
                return []
            pil_images = [_load_any_image(image) for image in images]
            embeds, embeds_ms = self._encode(pil_images)
            n_images = len(pil_images)
        n_images = embeds.shape[0]

        text_config = self.model.config.text_config
        decoder_start = int(text_config.bos_token_id)
        decoder = self.model.text_decoder
        decoder.generation_config.eos_token_id = int(
            getattr(text_config, "sep_token_id", None) or text_config.eos_token_id
        )
        decoder.generation_config.pad_token_id = int(text_config.pad_token_id)
        decoder.generation_config.bos_token_id = decoder_start

        started = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and self.device.type == "cuda"):
            output = decoder.generate(
                input_ids=torch.full((n_images, 1), decoder_start, dtype=torch.long, device=self.device),
                encoder_hidden_states=embeds,
                encoder_attention_mask=torch.ones(embeds.shape[:-1], dtype=torch.long, device=self.device),
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        total_ms = (time.time() - started) * 1000

        results: list[Caption] = []
        per_image_ms = total_ms / max(n_images, 1)
        embeds_share = embeds_ms / max(n_images, 1)
        for row in output:
            ids = row.tolist()
            if ids and ids[0] == decoder_start:
                ids = ids[1:]
            results.append(
                Caption(
                    text=self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    latency_ms=per_image_ms + embeds_share,
                    image_embeds_ms=embeds_share,
                )
            )
        return results


__all__ = ["BlipCaptioner", "Caption", "find_checkpoint"]
