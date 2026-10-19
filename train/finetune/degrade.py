"""Synthetic degradation, drawn from the flaw distribution VizWiz annotates.

The measured weakness of the captioner is not object vocabulary - it names
objects well - it is **unusable photographs**. On the 543 unrecognizable
validation images captioning collapses by roughly 4x (BLEU-4 0.0500 against
0.2089, 4-gram match 7.6% against 40.7%).

No public corpus contains natural blind-user blur, so it has to be generated.
This module applies the flaw types VizWiz actually annotates, with their real
frequencies as the sampling prior:

| code | flaw | positives in VizWiz train |
|---|---|---|
| BLR | blur | 2,240 |
| FRM | poor framing | 2,156 |
| ROT | rotated | 480 |
| BRT | too bright | 135 |
| DRK | too dark | 129 |
| OBS | obstructed | 116 |

Two deliberate properties:

**Degradation is applied to the image, before the vision tower.** It has to be:
the cached feature is the tower's output, so degrading features directly would
model something the tower never produces.

**It is seeded and reproducible.** A degraded cache must be rebuildable
byte-for-byte, or a training run cannot be explained afterwards.

The honest limit, which belongs in any write-up: this teaches the model to
caption degraded images *better*. It does not teach "this image is unusable".
That judgement is the quality gate's job, and the gate already exists.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

# Sampling weights, from the per-image flaw annotations in VizWiz-QualityIssues.
# Counts are the positives reported in docs/status-report.md.
FLAW_WEIGHTS = {
    "blur": 2240,
    "framing": 2156,
    "rotate": 480,
    "bright": 135,
    "dark": 129,
    "obstruct": 116,
}


@dataclass(frozen=True)
class DegradeConfig:
    """How aggressively to degrade, and how often."""

    probability: float = 0.5
    """Chance a given image is degraded at all. 0.5 gives the 50/50 clean/degraded
    mix that stops the model trading clean performance for degraded performance."""

    severity: float = 1.0
    """Multiplier on every transform's strength. 1.0 is calibrated against the
    VizWiz flaw distribution; higher values push further into unusability."""

    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "probability": self.probability,
            "severity": self.severity,
            "seed": self.seed,
            "flaw_weights": FLAW_WEIGHTS,
        }


def _blur(image, rng: random.Random, severity: float):
    """Gaussian blur, or a directional smear for motion blur.

    Motion blur is built by averaging several shifted copies rather than with
    ``ImageFilter.Kernel``: PIL's C filter accepts only a narrow set of kernel
    sizes and rejects larger odd ones with "bad kernel size", so a 7x7 or 9x9
    motion kernel is not reliably available. Averaging shifted copies has no size
    limit and produces a more realistic hand-shake smear anyway.
    """
    from PIL import Image, ImageChops, ImageFilter

    if rng.random() < 0.35:
        length = max(2, int(rng.uniform(3, 9) * severity))
        angle = rng.uniform(0, 180)
        radians = angle * math.pi / 180.0
        dx = length * math.cos(radians)
        dy = length * math.sin(radians)

        accumulator = image.convert("RGB").copy()
        steps = 4
        for step in range(1, steps + 1):
            fraction = step / steps
            shifted = ImageChops.offset(
                image.convert("RGB"), int(dx * fraction), int(dy * fraction)
            )
            accumulator = Image.blend(accumulator, shifted, 1.0 / (step + 1))
        return accumulator

    radius = rng.uniform(1.0, 4.0) * severity
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _dark(image, rng: random.Random, severity: float):
    """Underexposure: scale luminance down, then let contrast compress."""
    from PIL import ImageEnhance

    factor = max(0.15, 1.0 - rng.uniform(0.35, 0.7) * severity)
    return ImageEnhance.Brightness(image).enhance(factor)


def _bright(image, rng: random.Random, severity: float):
    """Overexposure, which clips highlights rather than merely scaling them."""
    from PIL import ImageEnhance

    factor = 1.0 + rng.uniform(0.4, 1.1) * severity
    return ImageEnhance.Brightness(image).enhance(factor)


def _rotate(image, rng: random.Random, severity: float):
    """Small-to-moderate rotation. VizWiz rotation is usually near-axis."""
    from PIL import Image

    angle = rng.uniform(-18, 18) * severity
    return image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))


def _obstruct(image, rng: random.Random, severity: float):
    """Occlude part of the frame, as a finger or object in front of the lens."""
    from PIL import ImageDraw

    width, height = image.size
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(1, 3)):
        w = rng.uniform(0.15, 0.45) * width * severity
        h = rng.uniform(0.15, 0.45) * height * severity
        x = rng.uniform(0, width - w)
        y = rng.uniform(0, height - h)
        shade = rng.randint(0, 60)
        draw.rectangle([x, y, x + w, y + h], fill=(shade, shade, shade))
    return image


def _framing(image, rng: random.Random, severity: float):
    """Poor framing: crop off-centre and off-subject, then rescale back."""
    from PIL import Image

    width, height = image.size
    # Keep 60-85% of the frame, biased toward one edge.
    keep = rng.uniform(0.6, 0.85)
    cw, ch = int(width * keep), int(height * keep)
    x = rng.choice([0, max(0, width - cw), rng.randint(0, max(0, width - cw))])
    y = rng.choice([0, max(0, height - ch), rng.randint(0, max(0, height - ch))])
    cropped = image.crop((x, y, x + cw, y + ch))
    return cropped.resize((width, height), resample=Image.BICUBIC)


def _jpeg(image, rng: random.Random, severity: float):
    """Compression artefacts, which VizWiz photographs carry anyway."""
    from PIL import Image

    quality = int(max(15, 75 - 45 * severity * rng.random()))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


TRANSFORMS = {
    "blur": _blur,
    "dark": _dark,
    "bright": _bright,
    "rotate": _rotate,
    "obstruct": _obstruct,
    "framing": _framing,
}

# Every degraded image also gets a compression pass, because real phone
# photographs do. It is not a flaw category of its own in the annotation.
ALWAYS = ("jpeg",)


def degrade_image(image, config: DegradeConfig, key: str):
    """Apply a seeded degradation to one image.

    ``key`` makes the transform reproducible per image: the same image and the
    same config always produce the same degraded result, regardless of iteration
    order or worker count.
    """
    rng = random.Random(f"{config.seed}:{key}")

    if rng.random() >= config.probability:
        return image, None

    names = list(FLAW_WEIGHTS)
    weights = [FLAW_WEIGHTS[n] for n in names]
    chosen = rng.choices(names, weights=weights, k=1)[0]

    out = image
    # Compound with one extra flaw a third of the time, which is what real bad
    # photographs look like: blurry *and* dark.
    applied = [chosen]
    if rng.random() < 0.33:
        extra = rng.choices(names, weights=weights, k=1)[0]
        if extra != chosen:
            applied.append(extra)

    for name in applied:
        out = TRANSFORMS[name](out, rng, config.severity)
    for name in ALWAYS:
        out = _jpeg(out, rng, config.severity)

    return out, "+".join(applied)


__all__ = [
    "ALWAYS",
    "FLAW_WEIGHTS",
    "TRANSFORMS",
    "DegradeConfig",
    "degrade_image",
]
