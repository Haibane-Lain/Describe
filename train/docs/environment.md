# Verified training environment

Recorded from the actual development machine. Re-verify with
`py -3.12 -m train.smoke_test` after any driver, ROCm, or PyTorch change.

## Hardware

| Component | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT (RDNA4, `gfx1201`, 16 GB) |
| CPU | AMD Ryzen 7 7800X3D (8 cores / 16 threads) |
| RAM | 31.1 GB |
| OS | Windows 11 Pro, build 26200 |
| Free disk (C:) | 159 GB |

## Software

| Component | Value |
|---|---|
| Interpreter | `py -3.12` → `C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe` (3.12.0) |
| PyTorch | `2.9.1+rocmsdk20260116` |
| torchvision | `0.24.1+rocmsdk20260116` |
| ROCm runtime | `7.2.26024-f6f897bd3d` (HIP) |
| ROCm SDK wheels | `rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries-custom` all `7.2.0.dev0` |
| numpy / pillow / tqdm | 2.4.4 / 12.2.0 / 4.67.3 |
| Graphics driver | `32.0.31041.1004` (Adrenalin 26.8.1, 2026-08-17) |

**Python 3.14 cannot be used.** AMD publishes `cp312` wheels only, and `cp312`
is a hard ABI match. Always invoke the project through `py -3.12`.

## Verification result

```
device : cuda:0
name   : AMD Radeon RX 9070 XT
backend: rocm
```

`torch.cuda.is_available()` returns `True` — under ROCm, HIP is exposed through
the `torch.cuda` namespace, so no vendor branching is needed in the code.

## Driver note: no downgrade was required

AMD's ROCm 7.2 documentation lists graphics driver **26.1.1** (January 2026) as a
prerequisite. This machine runs **26.8.1** (August 2026), seven months newer, and
ROCm PyTorch works correctly on it. The documented prerequisite was therefore
not enforced in practice on this configuration.

Practical consequence: **do not update or roll back the graphics driver to
"match" AMD's documentation.** The environment is verified working as it stands.
Keep the current driver installer archived in case a future driver update breaks
ROCm — that, not a downgrade, is the realistic risk.

## Verified checks

All four gates pass on this machine (RX 9070 XT, ROCm 7.2, MIOpen disabled):

| Command | Result |
|---|---|
| `py -3.12 -m train.verify_decoders` | 11/11 — beam correctness and speed |
| `py -3.12 -m train.smoke_test` | 18/18 — device, loss decrease, checkpoint round-trip |
| `py -3.12 -m train.features.extract --synthetic` | 7/7 — encoder, cache format, reload |
| `py -3.12 -m train.e2e_test` | 22/22 — full pipeline on generated data |

## Fine-tuning dependencies (BLIP path)

Added for the fine-tuned captioner in [../finetune/](../finetune/). The training
stack itself was already present:

| Component | Value |
|---|---|
| `transformers` | 5.5.4 — ships `blip`, `blip_2`, `smolvlm`, `qwen*_vl` modeling |
| `datasets` | 5.1.0 |
| `accelerate` | 1.13.0 |
| `peft` | 0.21.2 — installed for LoRA; pure Python, ROCm-safe |
| `safetensors` | 0.7.0 |
| `bitsandbytes` | **not installed, and not installable here** — see below |

**Do not plan around 4-bit or QLoRA on this machine.** `bitsandbytes` ROCm
support targets Linux; there is no supported 4-bit path on Windows ROCm. Memory
is therefore managed with bf16, a frozen and cached vision tower, gradient
checkpointing, and LoRA — none of which need a new kernel stack. Measured
figures are in [../finetune/README.md](../finetune/README.md).

**Model weight formats differ per repo.** `Salesforce/blip-image-captioning-base`
ships only `pytorch_model.bin`; `-large` ships only `model.safetensors`.
`use_safetensors=True` *errors* rather than falling back, so the loader tries
safetensors first and then the default load. Both are cached under
`~/.cache/huggingface/hub` (989.7 MB and 989.8 MB).

### This shell has no writable TEMP, which blocks Python entirely

`import torch` fails in the sandboxed shell with:

```
FileNotFoundError: [Errno 2] No usable temporary directory found in
['C:\\Users\\Lain\\AppData\\Local\\Temp', 'C:\\WINDOWS\\Temp', ...]
```

`tempfile` reaches `torch.utils.data` via `dill`, so the failure appears as a
torch import error and looks like a broken install. It is not: no directory the
process may write to is on the candidate list. Point `TEMP` and `TMP` at a
writable directory before running anything:

```powershell
$env:TEMP = "$PWD\.tmp"; $env:TMP = "$PWD\.tmp"
py -3.12 -c "import torch; print(torch.__version__)"
```

## Evaluation metrics: SPICE is unavailable

`pycocoevalcap` 1.2 and `pycocotools` 2.0.11 are installed. Run
`py -3.12 -m train.check_metrics` to confirm. Current status:

| Metric | Status |
|---|---|
| BLEU-1–4 | works |
| METEOR | works |
| ROUGE-L | works |
| CIDEr-D | works |
| **SPICE** | **fails** |

SPICE shells out to a bundled `spice-1.0.jar` built against Stanford CoreNLP
3.6.0. That jar uses FST reflection which modern JDKs block:

```
java.lang.reflect.InaccessibleObjectException: Unable to make field
private final byte[] java.lang.String.value accessible:
module java.base does not "opens java.lang" to unnamed module
```

Java 26 is installed here; SPICE would need an older JDK plus `--add-opens`
flags. **Report SPICE as unavailable in the paper, not as zero.** Four of the
five official VizWiz metrics are reported, which is defensible; a fabricated
zero is not.

## Measured throughput

ResNet-101 feature extraction, batch 64, 224×224, `float16` output:

| Workload | Images | Time |
|---|---|---|
| Encoder throughput | — | **156 img/s** |
| Flickr30k | 31,783 | ~3.4 min |
| VizWiz (train + val + test) | 39,181 | ~4.2 min |

Decoder throughput (cached features, 32 images, `max_len=30`):

| Decoder | Per image | 8,000 test images |
|---|---|---|
| Greedy | 2.8 ms | 0.4 min |
| Beam-3 | 4.3 ms | 0.6 min |
| Beam-5 | 7.3 ms | 1.0 min |

Feature extraction is a ~8 minute job for the entire project, and a full
test-split evaluation is about a minute with beam search. Neither stage is a
schedule risk.

Batch 32 and 64 give essentially the same encoder throughput (158.9 vs
156.0 img/s), so the GPU is not the bottleneck — image decode is.

## Decoder notes

**`<end>` is masked until `min_len`.** Beam search and greedy decoding both mask
the `<end>` token until `min_len` tokens have been generated. Without the mask, a
`min_len` larger than the natural caption length forces the search to emit
`<end>` repeatedly to pad itself up to the minimum.

**Finished beams are retired.** Once a beam emits `<end>` it is removed from the
active set by masking its score to `-inf`, and the search stops when every beam
for an image has finished. Before this, a completed beam kept generating `<end>`
and padded its own sequence out to `max_len`, leaking `<end>` into the output as
`Brown brown brown <end> <end> ...`.

Both defects were caught by `verify_decoders`, which asserts that no `<end>`
appears in any output and that a deliberately over-large `min_len` produces no
`<end>` padding. The retirement change also made beam-5 roughly 40% faster,
because the search now exits as soon as all beams have finished.

Keep `min_len` well below the expected caption length — 2–3 for VizWiz, where
captions average roughly 10–12 tokens.

## Known issue: MIOpen dropout kernel fails on gfx1201

**Symptom.** A multi-layer `nn.LSTM` with inter-layer dropout (`dropout > 0`)
raises `RuntimeError: miopenStatusUnknownError` and can appear to hang while
MIOpen attempts a JIT compile:

```
MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6)
  .../miopen_rocrand.hpp:45:10: fatal error:
  'rocrand/rocrand_xorwow.h' file not found
  1 error generated when compiling for gfx1201.
```

**Cause.** MIOpen's fused RNN path JIT-compiles `MIOpenDropoutHIP.cpp`, which
includes `rocrand/rocrand_xorwow.h`. The ROCm 7.2 Windows SDK wheel does not
ship that header.

**Why it is easy to misdiagnose.** Only the *combination* fails. Verified
individually on this machine:

| Operation | Result |
|---|---|
| `nn.Dropout(0.5)` standalone | works |
| `nn.LSTM(num_layers=1)` (no internal dropout) | works |
| `nn.LSTM(num_layers=2, dropout=0.5)` | **fails** |
| `ResNet-101` convolution forward | works |

**Resolution.** MIOpen is disabled at runtime, routing RNN and convolution work
to PyTorch's native implementations. Implemented in
`train/utils/device.py::configure_for_speed()`, which applies this automatically
on RDNA4 parts (`gfx1200`/`gfx1201`) via the `runtime.miopen: auto` config key.

Verified working after the workaround:

- 2-layer LSTM with `dropout=0.5` — forward and backward pass
- ResNet-101 convolution — 0.41 s for a batch of 8 at 224×224

**Revisit when** AMD ships a ROCm SDK containing the rocrand headers. At that
point set `runtime.miopen: enable` and re-run the smoke test to confirm.

## Measured throughput

ResNet-101 feature extraction, batch 64, 224×224, `float16` output, MIOpen
disabled:

| Workload | Images | Time |
|---|---|---|
| Encoder throughput | — | **156 img/s** |
| Flickr30k | 31,783 | ~3.4 min |
| VizWiz (train + val + test) | 39,181 | ~4.2 min |

Feature extraction is therefore a ~8 minute job for the entire project, not the
multi-hour CPU task originally budgeted. Batch 32 and 64 give essentially the
same throughput (158.9 vs 156.0 img/s), so the GPU is not the bottleneck —
image decode is.

Decoder training reads cached features and never touches an image again, so the
whole pipeline is comfortably inside the schedule.

## Reproducing this environment from scratch

```powershell
# ROCm SDK components
pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

# PyTorch for ROCm
pip install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```

> **Never run a bare `pip install torch` in this environment.** It replaces the
> ROCm build with the default CPU build and silently disables the GPU.

## References

- [ROCm 7.2 Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) — lists `gfx1201` and PyTorch 2.9 with Python 3.12
- [PyTorch via PIP installation (Windows)](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html) — the wheel URLs above
- [TheRock supported GPUs](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) — nightly wheel index per architecture
