# Onboarding: run the Describe backend on an NVIDIA machine + Android phone

Everything needed to go from a fresh machine to a working demo. Written for an
**NVIDIA GPU + Android phone**, which differs from the machine this was developed
on (AMD Radeon, ROCm) in exactly one way: the PyTorch build. Nothing in the code
assumes AMD.

Total time: **~40 minutes**, of which ~25 is downloads.

---

## 0. What you need

| | |
|---|---|
| OS | Windows 10/11, or Linux (steps are identical except shell syntax) |
| GPU | NVIDIA, ≥ 4 GB VRAM (inference needs ~2 GB) |
| Python | **3.12** exactly — the wheels and pinned versions assume it |
| Disk | ~4 GB free (model weights ~1 GB + PyTorch ~2.5 GB) |
| Network | internet for the first run (HuggingFace downloads BLIP weights) |
| Phone | Android 7.0+ with USB debugging |

**You do NOT need:**
- Any training data, datasets, or feature caches. Serving needs none.
- The OCR stack. OCR is disabled by default; `easyocr` is not installed.
- An AMD GPU. The ROCm-specific code paths are inert on CUDA.

---

## 1. Clone the repo — **with Git LFS**

The model weights live in Git LFS (~640 MB total, of which 546 MB is the two
models you actually need). **If you clone without LFS you get 134-byte pointer
files instead of weights**, and the server will fail with a confusing checkpoint
error. Do not skip this step.

```bash
git lfs install
git clone https://github.com/Haibane-Lain/Describe.git
cd Describe
git lfs pull
```

Verify you have real files. `git lfs ls-files` lists four entries, each with a
`*` marker meaning the file is downloaded locally:

```
1978b4beda * checkpoints/blip_large_head/best.pt      544 MB  the captioner
20a0ee632c * checkpoints/quality_gate/best.pt           2 MB  the quality gate
4a5efbfb48 * checkpoints/easyocr/craft_mlt_25k.pth     79 MB  OCR (unused)
e2272681d9 * checkpoints/easyocr/english_g2.pth        14 MB  OCR (unused)
```

A `-` instead of `*` means that file is still a pointer. Sanity check the sizes:

```powershell
Get-ChildItem checkpoints -Recurse -Filter *.pt | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

If any file shows **0.1 MB or less**, LFS did not run — see
[Troubleshooting](#9-troubleshooting).

You can skip the unused OCR weights to save 94 MB:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Haibane-Lain/Describe.git
cd Describe
git lfs pull --include "checkpoints/blip_large_head/*,checkpoints/quality_gate/*"
```

---

## 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

On Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If PowerShell refuses with an execution-policy error:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

That affects only the current shell and reverts when you close it.

---

## 3. Install PyTorch for your GPU — **before anything else**

Install the build matching your CUDA version. Pick the right index URL from
<https://pytorch.org/get-started/locally/>. For CUDA 12.8:

```powershell
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

> ### Do this first, and check it before continuing
> If you install other packages first, a dependency resolver may "helpfully"
> replace your CUDA torch with a CPU build, silently disabling the GPU. This is
> the single most common way this setup goes wrong.
>
> ```powershell
> python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> ```
>
> Must print `True` and your GPU name. If it prints `False`, stop and fix this
> before going further — everything still runs, but ~10× slower on CPU.

---

## 4. Install the runtime dependencies

```powershell
pip install -r train/requirements-serve.txt
```

This is the **minimal** set for serving. It deliberately excludes the training
stack and OCR. Verified combination: torch 2.9.1, transformers 5.5.4, Python 3.12.

---

## 5. Verify the install

Run the built-in integration test. It generates synthetic clips, pushes them
through the real pipeline (frame sampling → gate → captioner → contract), and
checks the output:

```powershell
python -m train.serve_smoke --synthetic
```

Expected: **`17/17 checks passed`**.

`--synthetic` is the right mode on a fresh clone: the test normally uses a VizWiz
photograph from `data/raw/`, which is git-ignored and therefore absent. Without
the flag it falls back to `--synthetic` automatically and says so.

The status assertions (`ok` / `tooDark` / `lowConfidence`) are skipped in
synthetic mode, because a generated image is not a real scene. To exercise them,
point the test at a real photo — if you have the VizWiz dataset, the full suite
runs **23 checks**:

```powershell
python -m train.serve_smoke        # 23 checks when data/raw/ is present
```

Expected output includes:

```
  captioner  loaded
  gate       loaded
  ocr        disabled (unstable; pass --ocr to enable)
```

The **first run downloads BLIP-large (~1.5 GB) from HuggingFace** into
`~/.cache/huggingface`. Allow for that.

### Optional: confirm the dependency list is complete

If you changed anything in the pipeline, this re-derives the required packages by
running the real code and diffs them against the requirements file:

```powershell
python -m train._trace_serve_deps
```

Expected final line: *"every non-transitive distribution the pipeline loads is
listed"*. Torch and torchvision are reported separately because you install those
from the CUDA index.

---

## 6. Start the backend

```powershell
python -m train.serve --serve --port 8000
```

Wait for:

```
ready - gate threshold 0.235, 10 frames, ocr=off
listening on http://0.0.0.0:8000  (POST /describe, GET /health)
```

Find your LAN IP:

```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^127\.|^169\.254' } | Select-Object IPAddress, InterfaceAlias
```

Note the address (e.g. `192.168.1.42`). **The phone will need it.**

### Open the firewall

The phone cannot reach the PC unless port 8000 is allowed. **Admin PowerShell:**

```powershell
New-NetFirewallRule -DisplayName "describe backend 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

This is the **most common reason the app fails to connect.** If you skip it, the
app will say *"I cannot reach the describer"* and you will wrongly blame the app.

### Test it from the PC

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://192.168.1.42:8000/health     # your LAN IP — this is the phone's path
```

Both must return:

```json
{"ok":true,"gate_threshold":0.235,"n_frames":10,"ocr":false}
```

If the first works and the second does not, it is the firewall.

---

## 7. Install the app on the phone

### Get the APK

It is **not in the repo** (build artifacts are git-ignored). Two options:

**A. Ask the team for `describe-app-v1.0.0.zip`** (22.5 MB) — contains the 49.7 MB
APK, a checksum and a README. This is the fast path.

**B. Build it yourself** — see [Appendix: building the APK](#9-appendix-building-the-apk).
Budget ~1 hour the first time.

### Install

Enable **Developer options → USB debugging** on the phone, connect USB, and tap
**Allow** on the prompt.

```powershell
adb devices          # must show "<serial>  device", not "unauthorized"
adb install -r describe-app-release.apk
```

### Point the app at your backend

**This step is mandatory.** The app ships pointing at `http://10.0.2.2:8000`,
which is the **Android emulator** alias — it will not work on a physical phone.

In the app: **Settings → Describer address → Set address**, and enter your PC's
LAN IP:

```
http://192.168.1.42:8000
```

First, prove reachability from the phone itself: open that same URL **in the
phone's browser**. It must show the JSON from step 6. If it does not, the app
cannot work either.

---

## 8. Test the app

| # | Do | Expect |
|---|---|---|
| 1 | Launch app, say "Begin" | welcome spoken |
| 2 | Say **"Record or Capture"** (or pinch) | records 5 seconds |
| 3 | Hold still, point at a clear object | speaks a description, then *"You can say 'retake' to take another picture, or 'next' to continue."* |
| 4 | Say **"retake"** | returns to camera |
| 5 | Cover the lens, record again | status `tooDark` → *"Too dark. Try somewhere brighter…"* |
| 6 | Move the phone while recording | status `lowConfidence` → a blur/framing hint |
| 7 | **Stop the backend, record again** | speaks *"I cannot reach the describer at …"* and returns to camera — **must not hang** |

Test 7 is the one to check before demoing. A silent spinner in front of an
audience is the worst failure mode.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `checkpoints/...best.pt` is 134 bytes | LFS did not run. `git lfs install && git lfs pull` |
| `torch.cuda.is_available()` is `False` | CPU torch installed. Reinstall with the CUDA index URL (step 3) |
| *"I cannot reach the describer at …"* | Backend not running / wrong address / **firewall** (step 6) |
| App hangs with no audio | Backend died mid-request. Check the server console |
| `OSError: ... does not appear to have a file named model.safetensors` | Corrupt/partial LFS download — `git lfs pull` again |
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| `ModuleNotFoundError: multipart` | `pip install python-multipart` (needed for the upload) |
| First request takes 20 s | Expected. Models load once at startup; keep the server warm |
| Captions still contain "Text in the image reads:" | You are running an **old server**. Restart it; OCR is off by default |
| `adb devices` shows `unauthorized` | Unplug, replug, tap **Allow** on the phone |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | A build with a different key is installed: `adb uninstall com.example.describe_app` |

---

## 10. What the system does, and its limits

**Pipeline:** 5-second video → 10 sampled frames → quality gate picks the most
usable frame → BLIP-large captioner describes it → JSON to the app.

| Component | Status |
|---|---|
| Captioner (BLIP-large, fine-tuned) | BLEU-4 0.2034 on 6,042 held-out VizWiz images |
| Quality gate | false-accept 4.79%, AUC 0.964 |
| Frame selection | gate scores all 10 frames, best wins |
| OCR | **disabled** (unstable; re-enable with `--ocr`) |

**Limits you must not paper over:**

- **The captioner cannot read text.** It describes scenes and objects well but
  does not transcribe labels, titles or dates. **Never use its output for
  medication names, dosages or expiry dates** — that is explicitly out of scope.
- **There is no confidence score, by design.** The response carries `status`
  (`ok` / `lowConfidence` / `tooDark`) and nothing numeric. A percentage was tried
  and removed: measured against caption quality, sequence likelihood scored only
  Spearman 0.26 while correlating 0.53 with caption *length*, so any number built
  from it would have described verbosity rather than accuracy. Do not add one back
  without beating those numbers — see
  [train/docs/frontend-contract.md](../train/docs/frontend-contract.md).
- **OCR is off.** Descriptions carry no "Text in the image reads:" clause.
- **The gate is not infallible.** It bounds false accepts at ~4.8%, so roughly 1
  in 20 unusable photos may be described as if usable.

---

## 11. Key files

| Path | What |
|---|---|
| `train/serve.py` | **the backend** — CLI and HTTP |
| `train/serve_smoke.py` | integration test (23 checks) |
| `train/docs/frontend-contract.md` | the JSON contract the app depends on |
| `train/finetune/README.md` | how the captioner was trained and measured |
| `train/ocr/README.md` | the OCR component (currently disabled) |
| `FrontEnd Submission/describe/` | the Flutter app source |

---

## 12. Appendix: building the APK

Only needed if you are not sent the APK. Requires Flutter, Android SDK and
**JDK 17**.

> **Java 21+ breaks the build** with `Cannot invoke "java.io.File.exists()" because "parent" is null`.
> Use JDK 17 and point Flutter at it:
> ```powershell
> flutter config --jdk-dir "C:\path\to\jdk-17"
> ```

```powershell
git clone https://github.com/flutter/flutter.git -b stable C:\flutter
# add C:\flutter\bin to PATH, then:
flutter doctor            # fix anything it flags

cd "FrontEnd Submission\describe"
flutter pub get
flutter build apk --release
```

Output: `build/app/outputs/flutter-apk/app-release.apk` (~50 MB).

The release build is signed with Flutter's **debug key** (there is a TODO in
`android/app/build.gradle.kts`). Fine for sideloading and internal testing; it
**cannot be published to the Play Store** as-is.

---

## 13. Known gaps in this repo

Found while writing this guide — worth knowing so you are not surprised:

- **No APK in the repo**, by design (build artifacts are git-ignored). Ask the
  team for `describe-app-v1.0.0.zip`, or build it (§12).
- **`data/raw/` is git-ignored**, so a fresh clone has no VizWiz images. This is
  why `train.serve_smoke` needs `--synthetic`; the manifests themselves *are*
  tracked.
- **`train/requirements.txt` is the AMD/ROCm/training set.** Use
  `train/requirements-serve.txt` for serving on NVIDIA.
- The Flutter build warns that `camera_android_camerax` and `flutter_tts` apply
  the Kotlin Gradle Plugin in a way future Flutter versions will reject. Harmless
  today; upgrade those plugins eventually.
