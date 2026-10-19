"""Device selection and reproducibility helpers.

This project runs on an AMD Radeon RX 9070 XT (gfx1201) through the ROCm build
of PyTorch. ROCm is exposed through the ``torch.cuda`` namespace, so the code
below never branches on vendor - it only checks availability.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved compute device plus enough detail to record in a run log."""

    device: torch.device
    name: str
    backend: str
    torch_version: str
    runtime_version: str | None

    @property
    def is_gpu(self) -> bool:
        return self.device.type == "cuda"

    def describe(self) -> str:
        lines = [
            f"device        : {self.device}",
            f"name          : {self.name}",
            f"backend       : {self.backend}",
            f"torch         : {self.torch_version}",
            f"runtime       : {self.runtime_version or 'n/a'}",
        ]
        return "\n".join(lines)


def get_device(prefer_gpu: bool = True) -> DeviceInfo:
    """Resolve the best available device.

    ``torch.cuda.is_available()`` is True for both CUDA and ROCm builds; on this
    machine it reports the ROCm runtime backing the RX 9070 XT.
    """
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    backend = "rocm" if hip_version else ("cuda" if cuda_version else "cpu")

    if prefer_gpu and torch.cuda.is_available():
        index = torch.cuda.current_device()
        return DeviceInfo(
            device=torch.device("cuda", index),
            name=torch.cuda.get_device_name(index),
            backend=backend,
            torch_version=torch.__version__,
            runtime_version=hip_version or cuda_version,
        )

    return DeviceInfo(
        device=torch.device("cpu"),
        name="cpu",
        backend="cpu",
        torch_version=torch.__version__,
        runtime_version=None,
    )


def check_rocm_compatibility(info: DeviceInfo) -> list[str]:
    """Return warnings for known ROCm-on-Windows rough edges.

    The decoder is a small LSTM, so the known RDNA4 problem areas (Triton and
    flash-attention kernels) are not exercised. These checks exist to fail
    loudly rather than silently if the environment drifts.
    """
    warnings: list[str] = []
    if info.backend != "rocm":
        return warnings

    runtime = info.runtime_version or ""
    try:
        major = int(runtime.split(".")[0])
    except (ValueError, IndexError):
        major = 0

    if major and major < 7:
        warnings.append(
            f"ROCm runtime {runtime} is older than 7.x; gfx1201 support was "
            "added in ROCm 7.1. Reinstall the ROCm PyTorch wheels."
        )

    # torch.backends.cudnn maps to MIOpen under ROCm.
    if not torch.backends.cudnn.is_available():
        warnings.append(
            "MIOpen (torch.backends.cudnn) reports unavailable; convolutions "
            "will fall back to slower paths. Training still works."
        )

    return warnings


def set_seed(seed: int) -> None:
    """Seed every generator we rely on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_for_speed(miopen: str = "auto") -> dict[str, str]:
    """Apply the safe runtime settings for a small-model training loop.

    Args:
        miopen: ``"auto"`` (default), ``"disable"``, or ``"enable"``.
            ``auto`` disables MIOpen on RDNA4, because of the bug described in
            :func:`disable_miopen_for_rdna4`. Pass ``"enable"`` to override once
            AMD ships a ROCm SDK that includes the missing rocrand headers.

    Returns:
        The settings that were applied, for logging into a run record.
    """
    applied: dict[str, str] = {}

    torch.backends.cudnn.benchmark = True
    applied["cudnn.benchmark"] = "True"

    # TF32 does not exist on RDNA4; leave it off so behaviour is explicit.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            applied["matmul.allow_tf32"] = "False"
        except Exception:  # pragma: no cover - backend dependent
            pass

    resolved = miopen
    if miopen == "auto":
        resolved = "disable" if _is_rdna4() else "enable"

    if resolved == "disable":
        torch.backends.cudnn.enabled = False
        applied["cudnn.enabled"] = "False (MIOpen disabled)"
        # MIOpen is off, so its kernel-search mode is irrelevant.
    else:
        torch.backends.cudnn.enabled = True
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
        applied["cudnn.enabled"] = "True"
        applied["MIOPEN_FIND_MODE"] = os.environ["MIOPEN_FIND_MODE"]

    return applied


def _is_rdna4() -> bool:
    """True when the active device is an RDNA4 part (gfx1200/gfx1201)."""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
    except Exception:  # pragma: no cover - driver dependent
        return False
    return "9070" in name or "9060" in name or "r9700" in name


def disable_miopen_for_rdna4() -> str:
    """Explain and apply the MIOpen workaround for RDNA4 on Windows.

    Observed on this machine (RX 9070 XT, gfx1201, ROCm 7.2.26024):

        MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION
        .../miopen_rocrand.hpp:45:10: fatal error:
        'rocrand/rocrand_xorwow.h' file not found
        RuntimeError: miopenStatusUnknownError

    MIOpen's fused RNN path JIT-compiles ``MIOpenDropoutHIP.cpp``, which
    includes ``rocrand/rocrand_xorwow.h``. The ROCm SDK Windows wheel does not
    ship that header, so **any multi-layer LSTM with inter-layer dropout fails**
    on this configuration. Single-layer LSTMs and standalone ``nn.Dropout`` are
    unaffected, which is why the failure looks intermittent.

    Disabling MIOpen routes RNN and convolution work to PyTorch's native
    implementations. Verified working for both a 2-layer LSTM with dropout and
    ResNet-101 convolutions on this GPU.
    """
    torch.backends.cudnn.enabled = False
    return (
        "MIOpen disabled: ROCm 7.2 Windows SDK is missing rocrand headers "
        "required by the MIOpen dropout kernel on gfx1201."
    )
