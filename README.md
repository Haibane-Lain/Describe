# Describe

An offline image-description assistant for blind and low-vision users. The user
records a 5-second video on an Android phone; a local backend picks the most
usable frame, describes it, and the phone speaks the result.

**New here? Read [docs/onboarding-nvidia.md](docs/onboarding-nvidia.md).** It is
the complete path from a fresh machine to a working demo (~40 minutes), written
for an NVIDIA GPU + Android phone.

---

## How it fits together

```
Android app (Flutter)                Backend (Python, local GPU)
────────────────────                 ──────────────────────────
record 5 s video  ──── HTTP ────▶   sample 10 frames
                                     quality gate picks best frame
                                     BLIP-large captioner describes it
   speak the result  ◀── JSON ────   {"status", "text", "confidence", "tags", "hint"}
```

Everything runs locally. No image or video leaves the machine.

| Component | What it does | Measured |
|---|---|---|
| **Captioner** | BLIP-large fine-tuned on VizWiz | BLEU-4 **0.2034** on 6,042 held-out images |
| **Quality gate** | rejects unusable photos before captioning | FA **4.79%**, AUC **0.964** |
| **OCR** | printed text | **disabled** (unstable) |

---

## Quick start

```bash
git lfs install
git clone https://github.com/Haibane-Lain/Describe.git
cd Describe

python3.12 -m venv .venv && .venv/Scripts/activate     # Windows
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r train/requirements-serve.txt

python -m train.serve --serve --port 8000
```

Then install the Android app and point it at `http://<your-LAN-IP>:8000`.
Full details, firewall setup and troubleshooting: **[docs/onboarding-nvidia.md](docs/onboarding-nvidia.md)**.

> **The model weights are in Git LFS.** Cloning without `git lfs install` gives
> you 134-byte pointer files instead of ~640 MB of weights, and the server will
> fail with a checkpoint error.

---

## Important limits

- **The captioner cannot read text.** It describes scenes and objects, and does
  not transcribe labels, titles or dates. **Do not use its output for medication
  names, dosages or expiry dates** — that is explicitly out of scope.
- **`confidence` is a placeholder** in the JSON. The real signal is `status`:
  `ok` / `lowConfidence` / `tooDark`.
- **OCR is off by default** (re-enable with `--ocr`).

---

## Repository layout

| Path | |
|---|---|
| `train/serve.py` | the backend — CLI and HTTP |
| `train/finetune/` | captioner training, evaluation and prediction |
| `train/quality/` | the quality gate |
| `train/ocr/` | OCR component (currently disabled) |
| `FrontEnd Submission/describe/` | the Flutter app |
| `train/docs/` | training process, experiments, contracts, OCR handoff |
| `docs/` | onboarding |

---

## Documentation

| Document | Read it if you want to… |
|---|---|
| [docs/onboarding-nvidia.md](docs/onboarding-nvidia.md) | **run it on your own machine and phone** |
| [train/docs/frontend-contract.md](train/docs/frontend-contract.md) | integrate the app with the backend |
| [train/docs/training-process.md](train/docs/training-process.md) | understand how the model was trained |
| [train/docs/captioner-experiments.md](train/docs/captioner-experiments.md) | know what was tried and failed, so you don't repeat it |
| [train/docs/ocr-plan.md](train/docs/ocr-plan.md) | work on the reading/text bucket |

---

## Hardware note

The project was developed on an **AMD Radeon (ROCm)** machine, but nothing
requires AMD. `train/utils/device.py` detects ROCm vs CUDA and only applies the
AMD-specific workarounds when a RDNA4 card is present. On NVIDIA the code runs
unmodified; the only difference is which PyTorch build you install.

Serving needs ~2 GB VRAM. Training needs more.
