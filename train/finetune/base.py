"""BLIP loading and the cached-feature forward pass.

The point of this module is one specific optimisation that makes fine-tuning
feasible on a single 16 GB Radeon.

``BlipForConditionalGeneration.forward`` does two things in sequence::

    image_embeds = self.vision_model(pixel_values=pixel_values).last_hidden_state
    outputs = self.text_decoder(
        input_ids=input_ids,
        encoder_hidden_states=image_embeds,
        labels=labels,
        reduction="mean",
    )

and ``BlipTextLMHeadModel.forward`` accepts ``encoder_hidden_states`` directly.
So when the vision tower is frozen, its 577 x 768 output for every training
image can be computed **once** and replayed for every epoch. Training then never
touches a JPEG or a convolution again.

This is the same trade the original ResNet + LSTM pipeline made (see
``train/features/extract.py``), applied to a model that actually works.

The wrapper below therefore presents a ``BlipForConditionalGeneration``-shaped
module whose forward accepts ``image_embeds`` instead of ``pixel_values``, while
keeping the original checkpoint's parameter prefixes (``vision_model.*``,
``text_decoder.*``) intact. That second property matters: it means an ordinary
``load_state_dict`` restores the pretrained weights with no key remapping, and a
support-only checkpoint (vision tower dropped) can still be loaded by prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DEFAULT_BASE_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_LARGE_MODEL = "Salesforce/blip-image-captioning-large"


@dataclass(frozen=True)
class LoadReport:
    """What actually came out of the pretrained checkpoint.

    Recorded into the run record so a run can be audited later without
    guessing which weights it started from.
    """

    model_id: str
    is_large: bool
    num_parameters_total: int
    num_parameters_text_decoder: int
    image_size: int
    num_image_tokens: int
    text_hidden_size: int
    vision_hidden_size: int
    fp32_layer_norm: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "is_large": self.is_large,
            "num_parameters_total": self.num_parameters_total,
            "num_parameters_text_decoder": self.num_parameters_text_decoder,
            "image_size": self.image_size,
            "num_image_tokens": self.num_image_tokens,
            "text_hidden_size": self.text_hidden_size,
            "vision_hidden_size": self.vision_hidden_size,
            "fp32_layer_norm": self.fp32_layer_norm,
        }


def vision_token_count(vision_model) -> int:
    """Number of image tokens the vision tower emits.

    In this transformers version `embeddings.position_embedding` is a bare
    `nn.Parameter` of shape ``[1, tokens, hidden]`` rather than an
    `nn.Embedding`, so accept both and read the token axis rather than guessing
    from the config. For the released checkpoints this is 577 (576 patches plus
    a CLS token).
    """
    embeddings = vision_model.embeddings
    position_embedding = getattr(embeddings, "position_embedding", None)
    if position_embedding is None:
        config = vision_model.config
        return int(config.image_size // config.patch_size) ** 2 + 1

    if hasattr(position_embedding, "weight"):  # nn.Embedding
        return int(position_embedding.weight.shape[1])
    return int(position_embedding.shape[1])  # nn.Parameter


class BlipCachedCaptioner(nn.Module):
    """BLIP captioner whose forward takes cached image features.

    Submodules are registered under the *same attribute names* as
    :class:`transformers.BlipForConditionalGeneration` (``vision_model`` and
    ``text_decoder``), so ``state_dict`` keys match the Hub checkpoint exactly.

    ``vision_model`` is optional. Once features are cached it is dead weight
    (~300 MB for BLIP-large) and must not be loaded into an optimizer, so it can
    be built or discarded independently via :meth:`supports_vision`.
    """

    def __init__(self, config, *, with_vision: bool = True, num_image_tokens: int | None = None) -> None:
        super().__init__()
        self.config = config

        if with_vision:
            from transformers.models.blip.modeling_blip import BlipVisionModel

            self.vision_model = BlipVisionModel(config.vision_config)
        else:
            self.vision_model = None

        from transformers.models.blip.modeling_blip_text import BlipTextLMHeadModel

        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

        # Authoritative image-token count, saved so it survives a round trip
        # through `state_dict` even when the vision tower is not loaded.
        if num_image_tokens is not None:
            self.num_image_tokens = int(num_image_tokens)

    # ------------------------------------------------------------------ helpers

    @property
    def supports_vision(self) -> bool:
        return self.vision_model is not None

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Delegate to the text decoder.

        Needed for ``peft`` to attach an adapter: its wrapping logic calls this
        on the top-level module, and an ``AttributeError`` here aborts LoRA
        setup with a message that points at the wrong thing entirely.
        """
        return self.text_decoder.prepare_inputs_for_generation(input_ids, **kwargs)

    def can_generate(self) -> bool:
        return True

    @property
    def image_embed_shape(self) -> tuple[int, int]:
        """``(num_image_tokens, hidden)`` expected by the text decoder.

        This is **577**, not 576, for a 384px/16px input: BLIP's vision tower
        prepends a CLS token to its 576 patch embeddings (verified against the
        released checkpoint: `position_embedding` is `[1, 577, 768]` and
        `last_hidden_state` is `[1, 577, 768]`).

        The count is read, never recomputed, because the two obvious formulas
        disagree with the checkpoint and with each other. Getting this wrong
        produces a cache that writes fine but cannot be broadcast into the
        decoder - which is exactly how the mistake surfaced, on real VizWiz
        images rather than in the synthetic smoke test.
        """
        cfg = self.config.vision_config
        hidden = getattr(cfg, "hidden_size", None)
        if hidden is None:  # pragma: no cover - defensive
            hidden = self.config.text_config.hidden_size

        if self.vision_model is not None:
            return vision_token_count(self.vision_model), hidden

        stored = getattr(self, "num_image_tokens", None)
        if stored is not None:
            return int(stored), hidden

        # Last resort. Correct for the shipped checkpoints.
        return int(cfg.image_size // cfg.patch_size) ** 2 + 1, hidden

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the vision tower once. Used only by the caching pass."""
        if self.vision_model is None:
            raise RuntimeError(
                "this instance was built without the vision tower "
                "(with_vision=False); use extract_features() to build one"
            )
        return self.vision_model(pixel_values=pixel_values).last_hidden_state

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        *,
        image_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ):
        """Forward with cached features.

        Pass exactly one of ``image_embeds`` (cached) or ``pixel_values``
        (uncached, requires the vision tower). The decoder shifts ``labels``
        internally, so pass unshifted caption ids.
        """
        if image_embeds is None:
            if pixel_values is None:
                raise ValueError("pass either image_embeds (cached) or pixel_values")
            image_embeds = self.encode_image(pixel_values)

        return self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            reduction="mean",
            logits_to_keep=logits_to_keep,
        )


# ---------------------------------------------------------------------- loading


def resolve_model_id(name: str) -> str:
    """Map a short name to a Hub id, passing through anything with a slash."""
    if "/" in name:
        return name
    aliases = {
        "base": DEFAULT_BASE_MODEL,
        "large": DEFAULT_LARGE_MODEL,
    }
    try:
        return aliases[name.lower()]
    except KeyError:
        raise SystemExit(
            f"unknown model '{name}'. Use one of {sorted(aliases)}, or pass a "
            "full Hugging Face repo id such as Salesforce/blip-image-captioning-base"
        ) from None


def build_model(
    model_id: str,
    *,
    with_vision: bool = True,
    fp32_layer_norm: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    """Load pretrained BLIP weights into :class:`BlipCachedCaptioner`.

    Args:
        with_vision: build the vision tower too (needed for the caching pass).
        fp32_layer_norm: keep ``LayerNorm`` in fp32, matching how the checkpoint
            was trained. Under bf16 autocast this is the safest setting; pass
            ``False`` only if it causes dtype friction on this backend.
        dtype: ``torch.float32`` by default. bf16 mixed precision belongs in the
            training loop's autocast context, not baked into the weights.

    Returns:
        ``(model, report)``.
    """
    from transformers import BlipConfig, BlipForConditionalGeneration

    dtype = torch.float32 if dtype is None else dtype

    # Build the stock model first: it carries the correct config with any
    # sub-config objects, and it is the thing the released weights were saved
    # from.
    #
    # `keep_in_fp32_modules` is a model *class* attribute in this transformers
    # version, not a from_pretrained kwarg (passing it as a kwarg raises
    # TypeError). Set it around the load and restore it afterwards so importing
    # this module has no lasting global effect.
    original_keep = getattr(BlipForConditionalGeneration, "_keep_in_fp32_modules", None)
    if fp32_layer_norm:
        BlipForConditionalGeneration._keep_in_fp32_modules = ["LayerNorm"]
    try:
        # weight format differs by repo: `Salesforce/blip-image-captioning-base`
        # ships only `pytorch_model.bin`, `-large` ships only
        # `model.safetensors`. `use_safetensors=True` does NOT fall back, it
        # errors, so try safetensors first and fall back to the default load.
        try:
            stock = BlipForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, use_safetensors=True
            )
        except (OSError, EnvironmentError):
            stock = BlipForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    finally:
        if fp32_layer_norm:
            if original_keep is None:
                del BlipForConditionalGeneration._keep_in_fp32_modules
            else:
                BlipForConditionalGeneration._keep_in_fp32_modules = original_keep
    config: BlipConfig = stock.config

    # Read the image-token count off the loaded checkpoint so it is authoritative
    # rather than recomputed by a formula.
    checkpoint_tokens = vision_token_count(stock.vision_model)

    wrapper = BlipCachedCaptioner(config, with_vision=with_vision, num_image_tokens=checkpoint_tokens)
    # Preserve the fp32 module dtype decisions made during loading (LayerNorm),
    # by copying the stock model's module dtypes rather than forcing one dtype
    # on every parameter.
    if fp32_layer_norm:
        stock_dtypes = {name: module.weight.dtype for name, module in stock.named_modules() if hasattr(module, "weight")}
        wrapper = _restore_module_dtypes(wrapper, stock_dtypes)

    # Ordinary state_dict load, no key remapping. `strict` is False so that a
    # vision-less wrapper can be built from the same checkpoint.
    missing, unexpected = wrapper.load_state_dict(stock.state_dict(), strict=False)
    del stock

    # Only vision_model.* may be missing (the with_vision=False case). Anything
    # else missing means the checkpoint and the config disagree - fail loudly
    # rather than training a half-random model.
    if with_vision:
        real_missing = list(missing)
    else:
        real_missing = [k for k in missing if not k.startswith("vision_model.")]
    if real_missing:
        raise RuntimeError(
            "pretrained weights did not load cleanly; missing keys: "
            f"{real_missing[:8]}{' ...' if len(real_missing) > 8 else ''}"
        )

    # Freeze everything by default. The training loop unfreezes exactly what the
    # chosen --tune mode wants. Without this, `--tune lora` would optimise the
    # entire 92 M-parameter text decoder plus the vision tower, which is neither
    # what the flag says nor what fits in memory.
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    num_tokens, hidden = wrapper.image_embed_shape
    report = LoadReport(
        model_id=model_id,
        is_large=bool(getattr(getattr(config, "vision_config", None), "num_hidden_layers", 12) >= 24),
        num_parameters_total=sum(p.numel() for p in wrapper.parameters()),
        num_parameters_text_decoder=sum(p.numel() for p in wrapper.text_decoder.parameters()),
        image_size=config.vision_config.image_size,
        num_image_tokens=num_tokens,
        text_hidden_size=hidden,
        vision_hidden_size=config.vision_config.hidden_size,
        fp32_layer_norm=fp32_layer_norm,
    )

    if device is not None:
        wrapper = wrapper.to(device)

    return wrapper, report


def _restore_module_dtypes(model: nn.Module, dtypes: dict[str, torch.dtype]) -> nn.Module:
    """Force each submodule back to the dtype it had in the source checkpoint.

    ``nn.Module.to(dtype)`` casts everything, which would undo the fp32
    LayerNorm kept by ``_keep_in_fp32_modules`` and produce a dtype mismatch
    between a fp32 LayerNorm and bf16 activations.
    """
    for name, module in model.named_modules():
        target = dtypes.get(name)
        if target is None or not hasattr(module, "weight"):
            continue
        if module.weight is not None and module.weight.dtype != target:
            module.to(target)
    return model


DEFAULT_LORA_LEAVES = frozenset({"query", "key", "value"})


def text_decoder_lora_targets(model: BlipCachedCaptioner | None = None) -> set[str]:
    """Leaf module names in the text decoder that are worth adapting.

    Returns leaf names (``query``, ``key``, ``value``, ...) rather than full
    paths on purpose: full paths would have to match the exact nesting of the
    wrapper, and getting that wrong fails silently by adapting *nothing*. This
    mistake was caught by the smoke test, which is why the function is shaped
    this way.

    The default set is the **attention projections only**. That is a deliberate
    choice, not minimalism for its own sake:

    * The text decoder's ``dense`` leaves include the attention output
      projection *and* both MLP projections (``intermediate.dense``,
      ``output.dense``). Adding ``dense`` triples the trainable count from
      2.4 M to 3.9 M and reached 193 img/s instead of 231 img/s in measurement,
      for no established benefit on captioning.
    * Adapting the attention queries and keys is what lets the decoder shift
      *where it looks* for the captioning domain, which is the part that
      transfers worst from COCO to a blind user's photograph.
    * The word embedding and the output projection are excluded outright: they
      are tied, they are the largest matrices in the decoder, and adapting them
      buys little while making the adapter heavy.
    """
    return set(DEFAULT_LORA_LEAVES)


def text_decoder_linear_count(model: BlipCachedCaptioner | None = None) -> set[str]:
    """Diagnostic helper: leaf names of the Linear layers actually present."""
    if model is None:
        return set()
    return {
        name.rsplit(".", 1)[-1]
        for name, module in model.text_decoder.named_modules()
        if isinstance(module, nn.Linear)
    }


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The tensors that are actually being optimised.

    Filters by *parameter* name, not by ``tensor.requires_grad``: a tensor
    pulled out of a ``state_dict`` is a detached leaf and always reports
    ``requires_grad=False``, so filtering on the tensor would silently save
    nothing at all.

    Support-only checkpoints stay small: with LoRA on the text decoder this is
    a few megabytes rather than the whole model, so a long history costs nothing
    to keep.
    """
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in trainable
    }


def _normalise_checkpoint_key(key: str, own_keys: set[str]) -> list[str]:
    """Candidate state-dict names for one key from a saved checkpoint.

    Keys come from three different writers and all of them must load:

    * plain (``trainable_state_dict``) -
      ``text_decoder.bert...query.weight``
    * peft on the text decoder -
      ``text_decoder.base_model.model.bert...query.lora_A.default.weight``
    * peft on the whole model -
      ``base_model.model.text_decoder.bert...query.lora_A.default.weight``

    So: drop the adapter-name segment (``default`` or an ``adapter``-prefixed
    token), drop the ``base_model``/``model`` wrapper segments, then try the
    result with and without a ``.weight``/``.bias`` suffix. Returning candidates
    rather than one answer lets the caller accept whichever name the model
    actually has.

    **Never drop a bare number.** Transformer layer indices are bare numbers,
    and treating them as adapter names silently rewrites ``layer.0.attention``
    to ``layer.attention``, which matches nothing.
    """
    parts = key.split(".")
    kept: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part == "base_model" and index + 1 < len(parts) and parts[index + 1] == "model":
            index += 2
            continue
        if part in {"base_model", "model"} or part == "default" or part.startswith("adapter"):
            index += 1
            continue
        kept.append(part)
        index += 1

    base = ".".join(kept)
    candidates = [base]
    for suffix in (".weight", ".bias"):
        candidates.append(base + suffix)
    # A trailing `.weight` that the model stores without it.
    for suffix in (".weight", ".bias"):
        if base.endswith(suffix):
            candidates.append(base[: -len(suffix)])
    return [c for c in candidates if c in own_keys] or [base]


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Load a support-only (or LoRA-wrapped) checkpoint into ``model``.

    Accepts the three key shapes described in :func:`_normalise_checkpoint_key`.
    Returns ``(loaded, skipped)`` so the caller can log exactly what happened.

    A load that matches nothing is the dangerous case here, not a load that
    errors: it leaves the pretrained model in place and reports the *base*
    model's metrics under the fine-tuned run's name. Callers must treat
    ``loaded`` being empty as a failure.
    """
    own = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []

    for key, value in state.items():
        candidates = _normalise_checkpoint_key(key, set(own))
        placed = False
        for candidate in candidates:
            if candidate in own and own[candidate].shape == value.shape:
                own[candidate] = value.to(own[candidate].dtype)
                loaded.append(candidate)
                placed = True
                break
        if not placed:
            skipped.append(key)

    if loaded:
        model.load_state_dict(own, strict=False)

    # Guard the failure mode that is worst here: adapter weights handed to a
    # model that has no adapter modules. Every key then resolves against a base
    # weight of the same shape, so the load "succeeds" by writing each weight
    # back onto itself, and the caller is told 144 tensors loaded while the
    # model is silently unadapted. Detected by checking whether the checkpoint
    # carries adapter tensors that the model cannot accept.
    adapter_in_state = [k for k in state if "lora_A" in k or "lora_B" in k]
    adapter_loaded = [k for k in loaded if "lora_" in k]
    if adapter_in_state and not adapter_loaded:
        raise RuntimeError(
            f"checkpoint holds {len(adapter_in_state)} LoRA tensors but none were applied; the "
            "model has no adapter modules attached. Call prepare_model_for_checkpoint() before "
            "loading, or the load silently leaves the model unadapted."
        )

    return loaded, skipped


def lora_rank_of(payload: dict) -> int:
    """Recover the LoRA rank from a checkpoint, for adapter reconstruction.

    The adapter must be rebuilt at exactly the trained rank before its weights
    can be loaded, and a rank mismatch is silent - the shapes simply do not
    match and every tensor is skipped.
    """
    config = payload.get("config") or {}
    if config.get("lora_rank"):
        return int(config["lora_rank"])
    for key, value in (payload.get("model") or {}).items():
        if "lora_A" in key:
            return int(value.shape[0])
    return 16


def tune_mode_of(payload: dict, default: str = "lora") -> str:
    """The ``--tune`` mode a checkpoint was produced with."""
    return (payload.get("config") or {}).get("tune", default)


def prepare_model_for_checkpoint(model: nn.Module, payload: dict, *, verbose: bool = True) -> str:
    """Put ``model`` into the shape its checkpoint expects, and return the tune mode.

    ``load_trainable_state_dict`` only restores parameters whose
    ``requires_grad`` is set, so the model must be unfrozen the way it was
    trained:

    * ``lora`` / ``lora-manual`` - attach the adapter modules at the trained
      rank. The adapter parameters are created trainable; the frozen base is
      already correct.
    * ``head`` - unfreeze the whole text decoder.

    Getting this wrong loads nothing, and the caller then either crashes or -
    far worse - reports the pretrained model's metrics under a fine-tuned run's
    name. Every consumer (evaluate, predict, verify) goes through this one
    function so the logic cannot drift between them.
    """
    tune = tune_mode_of(payload)

    if tune in ("lora", "lora-manual"):
        from train.finetune.train_blip import apply_manual_lora

        keys = list(payload.get("model", payload))
        uses_peft = any("base_model" in key for key in keys)
        rank = lora_rank_of(payload)
        if verbose:
            print(
                f"adapter    : {tune} r={rank}"
                + (" (peft-style keys)" if uses_peft else "")
                + ", attaching LoRA"
            )
        apply_manual_lora(model, rank=rank, alpha=2.0 * rank, dropout=0.0)
        return tune

    if tune == "head":
        parameters = model.text_decoder.parameters()
        expected = sum(p.numel() for p in parameters)
        if verbose:
            print(f"adapter    : head (full text decoder), unfreezing {expected / 1e6:.1f} M parameters")
        for parameter in parameters:
            parameter.requires_grad_(True)
        return tune

    raise SystemExit(f"unknown tune mode {tune!r} in the checkpoint config; cannot load it safely")


__all__ = [
    "DEFAULT_BASE_MODEL",
    "DEFAULT_LARGE_MODEL",
    "BlipCachedCaptioner",
    "LoadReport",
    "build_model",
    "load_trainable_state_dict",
    "lora_rank_of",
    "prepare_model_for_checkpoint",
    "resolve_model_id",
    "text_decoder_linear_count",
    "text_decoder_lora_targets",
    "trainable_state_dict",
    "tune_mode_of",
    "vision_token_count",
]
