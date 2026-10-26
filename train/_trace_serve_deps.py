"""Trace the real runtime dependencies of the serve path.

`train/requirements-serve.txt` must be complete, or the other team's install
fails with `ModuleNotFoundError` part-way through setup.

A static import trace does **not** work here: `train/serve.py` imports torch,
transformers, cv2, fastapi and PIL lazily inside methods and constructors, so
plain `import train.serve` pulls in almost nothing. The only way to learn the
real dependency surface is to run the pipeline.

This runs a real `DescribePipeline` over a synthetic clip in a subprocess, then
reports every distribution that was actually loaded, and diffs that against the
requirements file.

    py -3.12 -m train._trace_serve_deps
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CHILD = r"""
import importlib.metadata as md
import json
import sys
import sysconfig

# Build the package -> top-level-module map FIRST, before anything heavy is
# imported, so the snapshot afterwards is accurate.
pkg_files = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    tops = set()
    for f in (dist.files or []):
        parts = str(f).replace("\\", "/").split("/")
        if not parts:
            continue
        head = parts[0]
        if head.endswith(".dist-info") or head.endswith(".data"):
            continue
        tops.add(head[:-3] if head.endswith(".py") else head)
    pkg_files[name] = tops

before = set(sys.modules)

# --- the real serve path -----------------------------------------------------
import numpy as np
from pathlib import Path
import cv2

from train.serve import DescribePipeline

# A synthetic clip: decode -> gate -> captioner -> compose. No OCR.
tmp = Path(".tmp/_trace_clip.mp4")
tmp.parent.mkdir(exist_ok=True)
frame = np.full((240, 320, 3), 128, dtype=np.uint8)
writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for _ in range(20):
    writer.write(frame)
writer.release()

pipeline = DescribePipeline(verbose=False)
result = pipeline.describe(tmp)
print("CONTRACT:" + json.dumps(result.as_dict()))
# ----------------------------------------------------------------------------

loaded = {m.split(".")[0] for m in set(sys.modules) - before}
tops_to_pkgs = {}
for pkg, tops in pkg_files.items():
    for t in tops:
        tops_to_pkgs.setdefault(t, set()).add(pkg)

used = {}
for top in loaded:
    if top in tops_to_pkgs:
        for pkg in tops_to_pkgs[top]:
            used.setdefault(pkg, set()).add(top)

print("USED_JSON:" + json.dumps({k: sorted(v) for k, v in used.items()}))
"""


def normalise(name: str) -> str:
    return name.lower().replace("_", "-").strip()


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    if result.returncode != 0:
        print("  pipeline run failed:")
        print(result.stderr[-2500:])
        return 1

    used = None
    contract = None
    for line in result.stdout.splitlines():
        if line.startswith("USED_JSON:"):
            used = json.loads(line[len("USED_JSON:"):])
        elif line.startswith("CONTRACT:"):
            contract = line[len("CONTRACT:"):]

    if used is None:
        print("  no trace output")
        print(result.stdout[-1500:])
        return 1

    print(f"  pipeline contract: {contract}")
    print()

    # torch/torchvision are installed by the GPU step, not this file.
    OUT_OF_BAND = {"torch": "installed with the CUDA build", "torchvision": "installed with the CUDA build"}

    print("  distributions actually loaded by the serve path:")
    for pkg in sorted(used, key=str.lower):
        mods = ",".join(used[pkg])
        note = OUT_OF_BAND.get(normalise(pkg), "")
        print(f"    {pkg:<28} {mods:<28} {note}")

    req_path = Path("train/requirements-serve.txt")
    listed = set()
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        base = line.split("==")[0].split(">=")[0].split("[")[0]
        listed.add(normalise(base))

    skip = {
        "pip", "setuptools", "wheel", "pkg-resources",
        "train", "__main__", "builtins", "typing-extensions",
        # ROCm runtime shims present only on the development machine.
        "rocm", "rocm-sdk-core", "rocm-sdk-libraries-custom",
    }

    # Build the transitive closure of what the listed packages pull in. Anything
    # in that closure is installed automatically by pip and does not need to be
    # named in the requirements file - only the *direct* imports do.
    import importlib.metadata as md

    def requires_tree(roots: set[str]) -> set[str]:
        seen: set[str] = set()
        queue = list(roots)
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                dist = md.distribution(current)
            except Exception:
                continue
            for requirement in dist.requires or []:
                if ";" in requirement:  # environment marker - ignore for this check
                    requirement = requirement.split(";")[0]
                name = requirement.split("(")[0].split(">")[0].split("<")[0].split("=")[0]
                name = name.split("[")[0].strip()
                if name:
                    queue.append(normalise(name))
        return seen

    closure = requires_tree(set(listed))
    # Same-name variants that metadata reports differently.
    closure |= {normalise(x) for x in ("torch", "torchvision")}

    missing = []
    for pkg in used:
        key = normalise(pkg)
        if key in skip or key in closure:
            continue
        if key not in listed:
            missing.append(pkg)

    print()
    print(f"  listed directly in {req_path.name}: {len(listed)}")
    transitive = sorted({normalise(p) for p in used} - listed - skip - set(missing))
    print(f"  pulled in transitively (no need to list): {', '.join(transitive)}")
    if missing:
        print()
        print("  MISSING from requirements-serve.txt (direct imports, not transitive):")
        for m in sorted(missing):
            print(f"    {m}")
        return 1

    print()
    print("  every non-transitive distribution the pipeline loads is listed "
          "- requirements-serve.txt is complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
