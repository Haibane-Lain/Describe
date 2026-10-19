# Backend contract for the Flutter app

Everything here is implemented and verified. The Flutter app's existing
`DescribeResult.fromMap` schema is unchanged — this documents what the backend
now returns for it.

---

## Run the backend

```powershell
cd "C:\Coding\General Image Recognizer"
py -3.12 -m train.serve --serve --port 8000
```

- Loads the captioner, quality gate and OCR once at startup (~20 s). **Wait for
  `ready` before demoing** — do not let the first request pay the load cost.
- Listens on `0.0.0.0`, so it is reachable from other devices on the network.
- **Single worker by design.** EasyOCR's `readtext` is not thread-safe and the
  pipeline holds one model set. Do not run multiple workers or concurrent
  requests.

**Reaching it from the app:**

| Where the app runs | Base URL |
|---|---|
| Android emulator | `http://10.0.2.2:8000` |
| iOS simulator | `http://127.0.0.1:8000` |
| Physical phone on the same WiFi | `http://192.168.68.101:8000` |
| Desktop build | `http://127.0.0.1:8000` |

A phone **cannot** reach `localhost` — that resolves to the phone itself. This is
the single most likely cause of a failed demo, which is why the base URL is a
setting rather than a constant.

---

## `GET /health`

Use this to check the backend is up before recording.

```json
{"ok": true, "gate_threshold": 0.235, "n_frames": 10, "ocr": true}
```

---

## `POST /describe`

`multipart/form-data`, one field named **`video`** containing the recorded clip.

**Response** — exactly the map `DescribeResult.fromMap` already parses:

```json
{
  "status": "ok",
  "text": "a person is holding a bottle of vitamin water.",
  "confidence": 100,
  "tags": ["basic caption"],
  "hint": ""
}
```

### Fields

| Field | Type | Meaning |
|---|---|---|
| `status` | `"ok"` \| `"lowConfidence"` \| `"tooDark"` | matches the Dart `DescribeStatus` enum exactly |
| `text` | string | the caption |
| `confidence` | int | **always a placeholder** — see below |
| `tags` | list of string | currently always `["basic caption"]` |
| `hint` | string | retake advice; empty when `status` is `"ok"` |

### OCR is disabled

**`tags` will not contain `"text read"` and `text` will not contain a "Text in the
image reads:" clause.** OCR is currently unstable, so it is switched off and
nothing on the caption path depends on it. The model is kept — `train/ocr/`, its
frozen config and the `easyocr` weights are all intact — and the backend can
re-enable it with `--ocr`.

The app should not expect OCR text, and should not display or speak a
"text read" tag. If OCR comes back, `tags` will contain `"text read"` again and
the app needs no change, because it already renders whatever tags it is given.

### `status`

Derived from the quality gate, which was trained to bound its false-accept rate
at 4.79%:

- **`ok`** — the gate accepts the image. `text` is a real description.
- **`lowConfidence`** — the gate rejects it, and the dominant flaw is not
  darkness. `hint` names the flaw (blur, framing, occlusion).
- **`tooDark`** — the gate rejects it and the frame is genuinely dark
  (measured mean luminance, cross-checked against the gate's `DRK` head).
  `hint` advises more light.

The app already speaks a different warning per status, so this maps onto existing
UI with no change.

### `confidence` — deliberately a placeholder

**Do not display or rely on `confidence`.** It is `100` for `ok` and `0`
otherwise, purely because the Dart type requires a value.

Calibrated confidence is not implemented. What the model does is *lexical*
hedging — on unusable photographs it says "appears to be" or "too blurry" far more
often (75.5% of the worst images against 1.7% of good ones) — but that is phrase
counting, not a probability. Presenting it as a percentage would overstate what is
known. `status` is the honest signal and it is what the UI should use.

### `tags`

Currently always `["basic caption"]`, because OCR is disabled. The app renders
whatever tags it receives, so re-enabling OCR (`--ocr`) requires no app change.

When OCR is on, `"text read"` appears and its text is **text visible in the
image**, never verified fact — medication names, dosages and expiry dates are in
that bucket and the proposal excludes safety-critical use.

---

## Verified behaviour

Measured on this machine with real VizWiz photographs re-encoded as 5-second
clips:

| Input | `status` | `text` |
|---|---|---|
| normal photo | `ok` | "a person is holding a bottle of vitamin water." |
| near-black clip | `tooDark` | "…it's too dark to figure out what it is." |
| heavily blurred clip | `lowConfidence` | "a blurry picture of what appears to be a pill bottle." |

`hint` was populated correctly for both rejection cases.

**Latency: ~800 ms per request warm.** Cold start is ~20 s, once.

---

## What the app needs to change

Three things, all small:

1. **`processing.dart`** — replace the mocked `_askAgent()` with a real multipart
   POST to `/describe`, returning the same `Map<String, dynamic>` so
   `result.dart` is untouched.
2. **A base-URL setting** — `app_settings.dart` already exists; add a field.
   Default to `10.0.2.2` for Android emulators.
3. **`pubspec.yaml`** — add the `http` package.

**Unchanged:** every route, the `DescribeResult` schema, all accessibility
semantics, the TTS behaviour, and the three-status UI.

**One UX request.** `processing.dart` currently waits with a spinner and no
audio. For a blind user, silence during a ~1 s wait is worse than a spoken
"Describing". A short announcement would help; `Speaker.speak` is already
available.

---

## Error handling the app should implement

| Case | Expected app behaviour |
|---|---|
| Backend not running | speak a failure, never hang on the spinner |
| Timeout (>30 s) | speak a failure and offer retry |
| `status: "tooDark"` | speak `text` then `hint` (already implemented) |
| `status: "lowConfidence"` | speak `text` then the existing warning (already implemented) |

---

## Reproduce without the app

```powershell
# contract JSON straight from a clip
py -3.12 -m train.serve --video clip.mp4 --json

# or over HTTP
curl.exe -X POST -F "video=@clip.mp4" http://127.0.0.1:8000/describe
```

Useful while the app is being wired up: this lets the frontend team confirm the
payload without needing the app to build.
