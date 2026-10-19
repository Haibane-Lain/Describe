"""Integration: video in, caption + quality verdict out, in the frontend's contract shape.

This is the glue the project was missing. Two components are wired here:

    captioner   train.finetune.predict.BlipCaptioner   (BLIP-large, BLEU-4 0.2034)
    gate        train.quality.train_gate.QualityGate    (threshold 0.235, 8 flaw codes)

**OCR is disabled by default.** It is currently unstable, so nothing on the
caption path may depend on it. The model, its frozen config
(`train/configs/ocr_frozen.json`) and the whole `train/ocr/` package are kept
intact - only the call is switched off, and `--ocr` turns it back on. When it is
off, `tags` never contains `"text read"` and `text` never contains a "Text in the
image reads:" clause.

The Flutter app records a **5-second video** and expects a map with
``status``/``text``/``confidence``/``tags``/``hint``. Our models take a single
image, so the clip has to be reduced to one frame. That reduction is the one
piece of real work here, and the quality gate is the right tool for it: a blind
user cannot aim a camera, so a 5-second clip is several chances to get a usable
frame, and the gate can pick the best one.

``confidence`` is a deliberate placeholder. The app's Dart type requires the
field, but calibrated confidence is not implemented - what exists is *lexical*
hedging, which is not a probability. ``status`` carries the real signal, derived
from the gate's existing threshold and flaw heads.

CLI (Phase 1 - no frontend needed):

    py -3.12 -m train.serve --video clip.mp4
    py -3.12 -m train.serve --video clip.mp4 --ocr      # re-enable OCR
"""

# NOTE: `from __future__ import annotations` is deliberately NOT used here.
# It turns every annotation into a string, and FastAPI/Pydantic then cannot
# resolve `UploadFile` in the endpoint signature - it raises
# `PydanticUserError: class-not-fully-defined` at request time, producing a 500
# on every upload. The annotations in this module are all resolvable at runtime,
# so the future import is omitted on purpose. Do not add it back.

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Flaw codes from the gate's training formulation.
FLAW_HINTS = {
    "DRK": "Too dark. Try somewhere brighter, or turn on a light.",
    "BRT": "Too bright or washed out. Try moving away from the light.",
    "BLR": "Blurry. Hold the phone steadier and try again.",
    "ROT": "The photo is tilted. Try holding the phone level.",
    "FRM": "The subject is poorly framed. Try moving the phone closer or further away.",
    "OBS": "Something is blocking the view. Try moving the phone.",
    "OTH": "The photo is hard to make out. Try taking another one.",
    "NON": "",
}

# Status strings must match the Dart enum DescribeStatus exactly.
STATUS_OK = "ok"
STATUS_LOW = "lowConfidence"
STATUS_DARK = "tooDark"

TAG_CAPTION = "basic caption"
TAG_TEXT = "text read"

DEFAULT_FRAMES = 10

# Mean luminance (0-255) below which a frame is reported as too dark.
#
# The gate's flaw head does predict DRK, but only as the argmax for 1.6% of
# validation images, and it catches 53.5% of annotated-dark ones (100 of 187).
# Measured on a synthetic near-black clip it reported BLR instead - so routing
# `tooDark` from the flaw argmax alone would tell a user in the dark to hold the
# phone steadier. A direct luminance check is unambiguous for the one case where
# the remedy is "find more light", so it is used to confirm darkness and the flaw
# head supplies the remaining hints.
DARK_LUMINANCE = 45.0


@dataclass
class FrameScore:
    index: int
    position: float
    unrecognizable: float
    flaws: dict[str, float]

    @property
    def recognizable(self) -> bool:
        return False  # replaced by threshold check in the pipeline

    def top_flaw(self) -> str:
        if not self.flaws:
            return "OTH"
        return max(self.flaws.items(), key=lambda kv: kv[1])[0]


@dataclass
class DescribeOutput:
    """The contract the frontend consumes."""

    status: str
    text: str
    confidence: int
    tags: list[str] = field(default_factory=list)
    hint: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "text": self.text,
            "confidence": self.confidence,
            "tags": self.tags,
            "hint": self.hint,
        }


def sample_frames(video: Path | bytes, n: int = DEFAULT_FRAMES):
    """Decode ``n`` evenly spaced RGB frames from a video.

    Returns ``(frames, positions)`` where frames are HxWx3 RGB uint8 arrays and
    positions are the fractional offset of each frame in the clip. Sampling
    evenly rather than taking the first frame matters: the user is moving the
    phone, so early frames are often the worst.

    Bytes are written to a temporary file first. ``VideoCapture.open()`` does not
    accept a memory buffer for a container in this OpenCV build - it raises
    "Overload resolution failed" - and a real uploaded mp4 needs a seekable file
    for the container index anyway.
    """
    import cv2
    import numpy as np

    temporary: Path | None = None
    if isinstance(video, (bytes, bytearray)):
        import tempfile

        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            handle.write(video)
        finally:
            handle.close()
        temporary = Path(handle.name)
        source = str(temporary)
    else:
        source = str(video)

    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return [], []

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        positions = []
        if total > 0:
            wanted = [min(int(total * (i + 0.5) / n), total - 1) for i in range(n)]
            for target in wanted:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    positions.append(target / max(total, 1))
        else:
            # Stream with an unknown length: read it all, then thin to n.
            grabbed = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                grabbed.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if grabbed:
                step = max(1, len(grabbed) // n)
                frames = grabbed[::step][:n]
                positions = [i / max(len(frames), 1) for i in range(len(frames))]
        return frames, positions
    finally:
        capture.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DescribePipeline:
    """Loads every model once and turns a clip into the frontend contract."""

    def __init__(
        self,
        *,
        captioner_run: Path = Path("runs/blip_large_head"),
        gate_checkpoint: Path = Path("checkpoints/quality_gate/best.pt"),
        ocr_config: Path | None = None,
        n_frames: int = DEFAULT_FRAMES,
        num_beams: int = 5,
        device=None,
        verbose: bool = True,
    ) -> None:
        import torch

        from train.utils.device import configure_for_speed, get_device

        configure_for_speed()
        self.device = device if device is not None else get_device().device
        self.n_frames = n_frames
        self.num_beams = num_beams
        self.verbose = verbose
        self.timings: dict[str, float] = {}

        def log(message: str) -> None:
            if self.verbose:
                print(message, flush=True)

        started = time.time()
        from train.finetune.predict import BlipCaptioner

        self.captioner = BlipCaptioner.from_run(captioner_run, model="large", device=self.device)
        log(f"  captioner  loaded ({time.time() - started:.1f}s)")

        started = time.time()
        self._load_gate(gate_checkpoint)
        log(f"  gate       loaded ({time.time() - started:.1f}s)")

        # OCR is OFF by default. It is currently unstable, so the caption path must
        # not depend on it. The model, its frozen config and `train/ocr/` are all
        # left intact and reachable with --ocr; only the call is disabled.
        started = time.time()
        if ocr_config is not None:
            self._load_ocr(ocr_config)
            log(f"  ocr        ready  ({time.time() - started:.1f}s)")
        else:
            self.ocr_config = {}
            log("  ocr        disabled (unstable; pass --ocr to enable)")

    # ------------------------------------------------------------------ loading

    def _load_gate(self, checkpoint: Path) -> None:
        import torch

        from train.quality.train_gate import QualityGate

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config") or {}
        self.gate = QualityGate(
            feature_dim=int(config.get("feature_dim", 2048)),
            hidden=int(config.get("hidden", 256)),
            dropout=float(config.get("dropout", 0.3)),
        )
        missing, unexpected = self.gate.load_state_dict(payload["model_state"], strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"gate checkpoint does not match the model: missing={list(missing)} "
                f"unexpected={list(unexpected)}"
            )
        self.gate.eval().to(self.device)
        # The threshold bounds the false-accept rate; it is stored with the run.
        self.gate_threshold = float((payload.get("metrics") or {}).get("threshold", 0.235))
        self.flaw_codes = list(config.get("flaw_codes") or [])

    def _load_ocr(self, config_path: Path) -> None:
        from train.ocr import read_text as _read_text

        self._ocr_read = _read_text
        self.ocr_config = {}
        if config_path.exists():
            self.ocr_config = json.loads(config_path.read_text(encoding="utf-8"))
            self.ocr_config.pop("_provenance", None)

    # ------------------------------------------------------------------ stages

    def score_frames(self, frames: list) -> list[FrameScore]:
        """Run the gate over sampled frames to find the most usable one."""
        import numpy as np
        import torch

        from train.features.extract import build_encoder, preprocess

        if not frames:
            return []

        if not hasattr(self, "_encoder"):
            self._encoder, _ = build_encoder("resnet101")
            self._encoder = self._encoder.to(self.device).eval()

        size = 224  # the gate was trained on 224px pooled ResNet-101 features
        tensors = [preprocess(_to_pil(frame), size) for frame in frames]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            pooled = self._encoder(batch)
            recognizable_logit, flaw_logits = self.gate(pooled)
            unrecognizable = torch.sigmoid(recognizable_logit).cpu().numpy()
            flaws = torch.sigmoid(flaw_logits).cpu().numpy()

        scores = []
        for i in range(len(frames)):
            flaw_map = {
                code: float(flaws[i][j]) for j, code in enumerate(self.flaw_codes)
            } if self.flaw_codes else {}
            scores.append(
                FrameScore(
                    index=i,
                    position=0.0,
                    unrecognizable=float(unrecognizable[i]),
                    flaws=flaw_map,
                )
            )
        return scores

    def describe(self, video: Path | bytes) -> DescribeOutput:
        """Full pipeline: clip -> contract."""
        import numpy as np

        self.timings = {}
        started = time.time()

        frames, positions = sample_frames(video, self.n_frames)
        self.timings["decode"] = time.time() - started
        if not frames:
            return DescribeOutput(
                status=STATUS_LOW,
                text="I could not read the video. Please try again.",
                confidence=0,
                tags=[],
                hint="Try recording again, holding the phone steady.",
            )

        started = time.time()
        scores = self.score_frames(frames)
        for score, position in zip(scores, positions):
            score.position = position
        self.timings["gate"] = time.time() - started

        # Best frame = lowest unrecognizable probability. The gate routes to a
        # retake when it is NOT confident the image is fine, so lower is better.
        best = min(scores, key=lambda s: s.unrecognizable)
        best_frame = frames[best.index]

        started = time.time()
        caption = self._caption(best_frame)
        self.timings["caption"] = time.time() - started

        started = time.time()
        ocr_text, ocr_meta = self._ocr(best_frame)
        self.timings["ocr"] = time.time() - started

        return self._compose(best, best_frame, caption, ocr_text, ocr_meta)

    def _caption(self, frame) -> str:
        import numpy as np

        from train.finetune.predict import _preprocess

        import torch

        image = _to_pil(frame)
        size = self.captioner.model.config.vision_config.image_size
        pixel_values = _preprocess([image], size).to(self.device)
        with torch.no_grad():
            embeds = self.captioner.model.encode_image(pixel_values)
        captions = self.captioner.caption_batch(
            [image], num_beams=self.num_beams, max_new_tokens=30, image_embeds=embeds
        )
        return captions[0].text if captions else ""

    def _ocr(self, frame) -> tuple[str, dict]:
        import numpy as np

        if not self.ocr_config:
            return "", {}
        try:
            result = self._ocr_read(_to_pil(frame), **self.ocr_config)
        except Exception as exc:  # OCR must never break the caption path
            return "", {"error": str(exc)[:200]}
        text = (getattr(result, "text", "") or "").strip()
        return text, {
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
            "low_confidence": bool(getattr(result, "low_confidence", False)),
            "n_regions": int(getattr(result, "n_regions", 0) or 0),
        }

    # ---------------------------------------------------------------- composing

    def _compose(
        self, best: FrameScore, frame, caption: str, ocr_text: str, ocr_meta: dict
    ) -> DescribeOutput:
        import numpy as np

        caption = (caption or "").strip()
        tags = [TAG_CAPTION] if caption else []

        # Mean luminance of the frame actually chosen, used to confirm darkness.
        grey = np.asarray(_to_pil(frame).convert("L"), dtype=np.float32)
        luminance = float(grey.mean())

        # Status comes from the gate's own decision rule. It routes to a retake
        # when the unrecognizable probability is at or above the stored
        # threshold, so that is exactly the boundary used here.
        if best.unrecognizable >= self.gate_threshold:
            flaw = best.top_flaw()
            if luminance < DARK_LUMINANCE or flaw == "DRK":
                status = STATUS_DARK
                hint = FLAW_HINTS["DRK"]
            else:
                status = STATUS_LOW
                hint = FLAW_HINTS.get(flaw) or FLAW_HINTS["OTH"]
        else:
            status = STATUS_OK
            hint = ""

        pieces = []
        if caption:
            pieces.append(caption if caption.endswith((".", "!", "?")) else caption + ".")
        if ocr_text:
            spoken = ocr_text.rstrip(".")
            pieces.append(f"Text in the image reads: {spoken}.")
            tags.append(TAG_TEXT)
        if not pieces:
            pieces.append("I could not make out anything in this photo.")
            if not hint:
                hint = "Try taking another photo, holding the phone steady."

        return DescribeOutput(
            status=status,
            text=" ".join(pieces),
            confidence=100 if status == STATUS_OK else 0,
            tags=tags,
            hint=hint,
        )


def _to_pil(frame):
    from PIL import Image

    return frame if isinstance(frame, Image.Image) else Image.fromarray(frame)


# --------------------------------------------------------------------- HTTP API
#
# Phase 2. The models are loaded once at startup, not per request: cold start is
# ~20 s and a per-request load would make the demo unusable.
#
# Run with a SINGLE worker. `Reader.readtext` is not thread-safe (see
# train/ocr/engine.py) and the pipeline holds one captioner and one gate, so
# concurrent requests would share mutable state.
#
#     py -3.12 -m train.serve --serve --host 0.0.0.0 --port 8000
#
# The app cannot reach `localhost` from a phone. Use the dev machine's LAN IP, or
# 10.0.2.2 from an Android emulator.

_PIPELINE: "DescribePipeline | None" = None
_PIPELINE_LOCK = None


def build_app(pipeline: "DescribePipeline"):
    """Wrap a loaded pipeline in a FastAPI app."""
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="describe", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "gate_threshold": pipeline.gate_threshold,
            "n_frames": pipeline.n_frames,
            "ocr": bool(pipeline.ocr_config),
        }

    @app.post("/describe")
    async def describe(video: UploadFile = File(...)) -> JSONResponse:
        data = await video.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        try:
            result = pipeline.describe(data)
        except Exception as exc:  # never leak a stack trace to the client
            raise HTTPException(status_code=500, detail=f"describe failed: {exc}") from exc
        return JSONResponse(result.as_dict())

    return app


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Describe a video clip (caption + quality gate).")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print only the contract JSON")
    parser.add_argument("--save-frames", type=Path, default=None,
                        help="write the sampled frames to this directory for inspection")
    parser.add_argument("--ocr", action="store_true",
                        help="enable OCR (currently unstable and OFF by default; the model is "
                             "kept in train/ocr/ and this only controls whether it is called)")
    parser.add_argument("--serve", action="store_true", help="run the HTTP service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    ocr_config = Path("train/configs/ocr_frozen.json") if args.ocr else None

    if args.serve:
        import uvicorn

        print("loading models (this takes ~20 s)...", flush=True)
        pipeline = DescribePipeline(
            n_frames=args.frames,
            num_beams=args.beams,
            ocr_config=ocr_config,
            verbose=False,
        )
        print(f"ready - gate threshold {pipeline.gate_threshold:.3f}, "
              f"{pipeline.n_frames} frames, ocr={'on' if pipeline.ocr_config else 'off'}")
        print(f"listening on http://{args.host}:{args.port}  (POST /describe, GET /health)")
        # One worker: the pipeline is stateful and EasyOCR is not thread-safe.
        uvicorn.run(build_app(pipeline), host=args.host, port=args.port, workers=1, log_level="info")
        return 0

    if args.video is None:
        parser.error("--video is required unless --serve is given")
    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    pipeline = DescribePipeline(
        n_frames=args.frames,
        num_beams=args.beams,
        ocr_config=ocr_config,
        verbose=not args.json,
    )

    if args.save_frames:
        from PIL import Image

        args.save_frames.mkdir(parents=True, exist_ok=True)
        frames, positions = sample_frames(args.video, args.frames)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(args.save_frames / f"frame_{i:02d}.png")
        if not args.json:
            print(f"  saved {len(frames)} frames -> {args.save_frames}")

    started = time.time()
    result = pipeline.describe(args.video)
    elapsed = time.time() - started

    if args.json:
        print(json.dumps(result.as_dict()))
    else:
        print()
        print(f"  status     : {result.status}")
        print(f"  text       : {result.text}")
        print(f"  tags       : {result.tags}")
        if result.hint:
            print(f"  hint       : {result.hint}")
        print(f"  confidence : {result.confidence}  (placeholder - not implemented)")
        print()
        for stage, seconds in pipeline.timings.items():
            print(f"    {stage:<8} {seconds * 1000:>7.0f} ms")
        print(f"    {'TOTAL':<8} {elapsed * 1000:>7.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
