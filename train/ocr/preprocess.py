"""Image preparation for OCR: load, upscale, correct perspective, deskew.

``train/docs/build-plan.md`` §2.4 asks for EasyOCR "behind perspective
correction, deskewing, and upscaling". Each of those is a *conditional* step
here, not an unconditional pipeline: on a clean, upright, large photograph every
transform is a no-op, and running them anyway would blur text and cost time for
nothing. The flags recorded in :class:`PreprocessResult` say which fired, so the
measurement can attribute a hit-rate change to a specific step.

Why upscaling at all: measured on the 577 validation images, the long side is
p10 648 px, median 1,296 px, max 2,592 px, and **61 of 577 images are under
1,000 px**. CRAFT detects small dense text poorly below roughly that size, and
signs and labels are the two dominant subtypes.

Everything works on BGR ``uint8`` numpy arrays, because that is what EasyOCR
assumes when it is handed a 3-channel array (``easyocr.utils.reformat_input``
treats shape ``(h, w, 3)`` as BGR). Passing RGB silently swaps the channels,
which does not crash -- it just reads slightly worse. An explicit convention
beats a silent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Named constants rather than magic numbers: every one of these ends up in the
# run config so a measurement can be reproduced.
UPSAMPLE_LONG_SIDE = 1000
MIN_SKEW_DEG = 1.5
MAX_SKEW_DEG = 15.0
MIN_QUAD_AREA_FRAC = 0.40
MIN_TEXT_PIXEL_FRACTION = 0.005


@dataclass(frozen=True)
class PreprocessConfig:
    """Which preparation steps are allowed to run."""

    upscale: bool = True
    min_long_side: int = UPSAMPLE_LONG_SIDE
    deskew: bool = True
    min_skew_deg: float = MIN_SKEW_DEG
    max_skew_deg: float = MAX_SKEW_DEG
    perspective: bool = True
    min_quad_area_frac: float = MIN_QUAD_AREA_FRAC
    enhance: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "upscale": self.upscale,
            "min_long_side": self.min_long_side,
            "deskew": self.deskew,
            "min_skew_deg": self.min_skew_deg,
            "max_skew_deg": self.max_skew_deg,
            "perspective": self.perspective,
            "min_quad_area_frac": self.min_quad_area_frac,
            "enhance": self.enhance,
        }


DISABLED = PreprocessConfig(
    upscale=False,
    deskew=False,
    perspective=False,
    enhance=False,
)


@dataclass
class PreprocessResult:
    """A prepared image plus the mapping back to the original coordinates."""

    image: np.ndarray
    matrix: np.ndarray | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    @property
    def original_shape(self) -> tuple[int, int] | None:
        shape = self.flags.get("original_shape")
        return tuple(shape) if shape else None

    def to_original(self, points: list[list[float]] | np.ndarray) -> list[list[float]]:
        """Map points from the prepared image back to original pixel space."""
        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.matrix is None:
            return arr.tolist()
        try:
            inverse = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate matrix
            return arr.tolist()
        homogeneous = np.hstack([arr, np.ones((arr.shape[0], 1))])
        mapped = homogeneous @ inverse.T
        mapped = mapped[:, :2] / np.clip(mapped[:, 2:3], 1e-9, None)
        return mapped.tolist()


def load_image(source: Any) -> np.ndarray:
    """Read ``source`` into a BGR ``uint8`` array.

    Accepts a path, bytes, a PIL image, or an array. EXIF orientation is applied
    first: VizWiz photographs come from phones held in any orientation, and a
    sideways photograph is a needless OCR failure.
    """
    import cv2

    def _from_pil(im) -> np.ndarray:
        from PIL import Image, ImageOps

        if not isinstance(im, Image.Image):  # pragma: no cover - defensive
            raise TypeError(f"unsupported image object {type(im)!r}")
        im = ImageOps.exif_transpose(im)
        if im.mode == "L":
            return cv2.cvtColor(np.array(im), cv2.COLOR_GRAY2BGR)
        if im.mode == "RGBA":
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

    if isinstance(source, (str, Path)):
        from PIL import Image

        with Image.open(source) as im:
            return _from_pil(im)

    if isinstance(source, (bytes, bytearray)):
        import io

        from PIL import Image

        with Image.open(io.BytesIO(source)) as im:
            return _from_pil(im)

    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.astype(np.uint8)  # documented as BGR, passed through
        raise ValueError(f"unsupported array shape {arr.shape}")

    # Anything with a PIL-like interface (PIL.Image.Image subclasses included).
    return _from_pil(source)


def _text_mask(bgr: np.ndarray) -> np.ndarray:
    """Otsu mask with the strokes white, whichever way the image is lit."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # On a dark photograph the inversion flips the polarity; if the "text" covers
    # most of the frame it is the background, so flip it back.
    if float(mask.mean()) / 255.0 > 0.5:
        mask = cv2.bitwise_not(mask)
    return mask


def estimate_skew_deg(bgr: np.ndarray) -> float:
    """Skew correction, in degrees, to apply to ``bgr`` (empty images give 0)."""
    import cv2

    mask = _text_mask(bgr)
    if float((mask > 0).mean()) < MIN_TEXT_PIXEL_FRACTION:
        return 0.0

    height, width = mask.shape[:2]
    # Merge characters into lines before fitting, or the minimum-area rectangle
    # is dominated by individual glyphs and the angle means nothing.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 60), 3))
    merged = cv2.dilate(mask, kernel, iterations=1)
    coords = np.column_stack(np.where(merged > 0))
    if coords.shape[0] < 120:
        return 0.0

    angle = float(cv2.minAreaRect(coords)[-1])
    # OpenCV's angle convention has changed across versions; normalise to
    # "degrees to rotate back", in (-45, 45].
    if angle < -45.0:
        angle = -(90.0 + angle)
    else:
        angle = -angle
    return angle


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[int(np.argmin(sums))]
    ordered[2] = points[int(np.argmax(sums))]
    ordered[1] = points[int(np.argmin(diffs))]
    ordered[3] = points[int(np.argmax(diffs))]
    return ordered


def find_page_quad(bgr: np.ndarray, min_area_frac: float = MIN_QUAD_AREA_FRAC):
    """A plausible four-corner document/label/screen quad, or ``None``.

    Deliberately conservative: a wrong perspective warp destroys text, and a
    missed one only leaves the image as it was.
    """
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(bgr.shape[0] * bgr.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(contour) < min_area_frac * frame_area:
            break
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_quad(approx.reshape(4, 2))
    return None


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 3):
        return matrix.astype(np.float64)
    out = np.eye(3, dtype=np.float64)
    out[:2, :] = matrix
    return out


def prepare(image: Any, cfg: PreprocessConfig | None = None) -> PreprocessResult:
    """Load and conditionally prepare an image for OCR."""
    import cv2

    cfg = cfg or PreprocessConfig()
    bgr = load_image(image)
    original_shape = (int(bgr.shape[0]), int(bgr.shape[1]))
    matrix = np.eye(3, dtype=np.float64)
    flags: dict[str, Any] = {
        "original_shape": original_shape,
        "upscaled": False,
        "deskew_deg": 0.0,
        "perspective": False,
        "enhance": False,
    }

    if cfg.upscale:
        long_side = max(bgr.shape[0], bgr.shape[1])
        if long_side < cfg.min_long_side:
            scale = cfg.min_long_side / float(max(long_side, 1))
            bgr = cv2.resize(
                bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            matrix = _as_3x3(np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float64)) @ matrix
            flags["upscaled"] = True
            flags["scale"] = round(scale, 4)

    if cfg.perspective:
        quad = find_page_quad(bgr, cfg.min_quad_area_frac)
        if quad is not None:
            width = max(
                np.linalg.norm(quad[2] - quad[3]), np.linalg.norm(quad[1] - quad[0])
            )
            height = max(
                np.linalg.norm(quad[1] - quad[2]), np.linalg.norm(quad[0] - quad[3])
            )
            target = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
            homography = cv2.getPerspectiveTransform(quad, target)
            bgr = cv2.warpPerspective(
                bgr,
                homography,
                (max(int(width), 1), max(int(height), 1)),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            matrix = _as_3x3(homography) @ matrix
            flags["perspective"] = True

    if cfg.deskew:
        angle = estimate_skew_deg(bgr)
        flags["skew_estimate_deg"] = round(angle, 2)
        if cfg.min_skew_deg <= abs(angle) <= cfg.max_skew_deg:
            height, width = bgr.shape[:2]
            center = (width / 2.0, height / 2.0)
            rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
            new_w = int(height * sin + width * cos)
            new_h = int(height * cos + width * sin)
            rotation[0, 2] += new_w / 2.0 - center[0]
            rotation[1, 2] += new_h / 2.0 - center[1]
            bgr = cv2.warpAffine(
                bgr,
                rotation,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            matrix = _as_3x3(rotation) @ matrix
            flags["deskew_deg"] = round(angle, 2)

    if cfg.enhance:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lightness, a_chan, b_chan = cv2.split(lab)
        contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        bgr = cv2.cvtColor(cv2.merge([contrast, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
        flags["enhance"] = True

    flags["processed_shape"] = (int(bgr.shape[0]), int(bgr.shape[1]))
    identity_ish = np.allclose(matrix, np.eye(3))
    return PreprocessResult(
        image=bgr,
        matrix=None if identity_ish else matrix,
        flags=flags,
    )


def map_box_to_original(box, result: PreprocessResult) -> list[list[float]]:
    """Map a region box from the prepared image back to original pixels."""
    return result.to_original(box)
