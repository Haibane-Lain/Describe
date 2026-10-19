# Minimal Plan: Image Description App for Visually Impaired Users

**Team size:** 4 people | **Platform:** Mobile app | **Timeline:** 6 weeks | **SDGs:** 10 (Reduced Inequalities), 4 (Quality Education)

**Pitch:** A mobile app that takes a photo, checks whether it is usable, describes it by voice, and reads text aloud. It is an intelligent agent because it perceives (camera), decides (quality check and tool routing), and acts (spoken output or a retake request).

## 1. Stack decisions

| Layer | Choice | Why |
|---|---|---|
| Mobile app | React Native with Expo (Expo Go for testing) | One codebase for Android and iOS, no Android Studio setup, built-in camera and speech modules |
| Speech output | expo-speech (on-device TTS) | No server code, works offline, respects the phone's speech settings |
| Backend | Python + FastAPI | Models are in PyTorch, so the API lives in the same language |
| Models | PyTorch: CNN encoder + LSTM decoder (captioner), MobileNetV2 (quality classifier) | Matches the training plan |
| OCR | EasyOCR (pretrained, called as a tool) | Training OCR is out of scope |
| Image checks | OpenCV | Baseline quality rules |
| Logging | JSON/CSV files on the server | Enough for evaluation |
| Demo hosting | Laptop running FastAPI on the same Wi-Fi, or a tunnel (ngrok / Cloudflare Tunnel) | Free, with an easy backup plan |
| Training compute | RX 9070 XT if ROCm works within a day, otherwise Kaggle | Do not let drivers eat the schedule |

**Key decision:** inference runs on the server, on CPU. A small decoder takes about a second or two per image on CPU, which avoids ROCm risk at demo time and avoids converting models to run on the phone. On-device models (TFLite or ONNX) are a stretch goal.

## 2. Architecture

Phone camera → app uploads photo to `POST /describe` → server: quality classifier → (bad? return retake message) → router → captioner and/or OCR → composer → JSON response → app shows text and speaks it with expo-speech.

**API contract** (agree on this in week 1 so everyone can work in parallel):

```
POST /describe   (multipart: image, mode = "describe" | "read_text")
-> {
  "status": "ok" | "retake",
  "message": "A person holding a white bottle on a table.",
  "reason": "too_dark" | "blurry" | null,
  "confidence": 0.72,
  "tools_used": ["quality", "caption"],
  "latency_ms": 1450
}
```

## 3. Components

1. **Mobile app:** camera screen, two large buttons ("Describe" and "Read text"), a "Repeat" button, text result display, automatic speech of results.
2. **API server:** one endpoint, image validation, error handling.
3. **Quality classifier (trained):** fine-tuned MobileNetV2, compared against an OpenCV blur/brightness baseline.
4. **Captioner (trained):** frozen CNN encoder with cached features, LSTM decoder, trained on Flickr30k then fine-tuned on VizWiz-Captions, compared against pretrained BLIP.
5. **OCR tool:** EasyOCR wrapper returning text and confidence.
6. **Agent router:** rule-based. Quality fail → retake; "Read text" → OCR; otherwise caption; low confidence → hedged wording.
7. **Composer:** short, spoken-friendly sentences, warnings first.
8. **Logging:** timestamp, scores, tool used, output, latency.
9. **Evaluation set:** 30 to 50 images (VizWiz validation subset plus your own photos).

## 4. Agent description (PEAS)

| Element | Description |
|---|---|
| Performance measure | Description accuracy and usefulness, speed, correct retake decisions, user satisfaction |
| Environment | Real-world phone photos, often blurry, dark, or badly framed; partially observable and dynamic |
| Actuators | Speech output, on-screen text, spoken retake instructions |
| Sensors | Phone camera (microphone is a stretch goal) |

**Agent type:** goal-based with simple utility (confidence thresholds decide whether to describe, hedge, or ask for a retake).

## 5. Work assignments

| Person | Role | Owns | Deliverables |
|---|---|---|---|
| 1 | Captioning lead | Data pipeline, vocabulary, feature extraction, training, caption metrics | Trained captioner checkpoint, `caption.py` with `generate(image) -> text`, BLEU/CIDEr results vs. pretrained BLIP |
| 2 | Vision tools | Quality classifier, OpenCV baseline, OCR wrapper | Trained quality model, `quality.py`, `ocr.py`, comparison of rules vs. learned model |
| 3 | Backend and agent | FastAPI server, router, composer, logging, demo deployment | Working `/describe` endpoint (stub models in week 1), `agent.py`, `composer.py`, logs, agent-level tests |
| 4 | Mobile app and QA | Expo app, camera, TTS, accessibility testing, user testing, report coordination | Working app, screen reader (TalkBack or VoiceOver) test notes, 5 to 10 user tests, report structure and slides |

**Report sections:** everyone writes their own section (method, results, limitations). Persons 1 and 2 write the ML sections, Person 3 writes architecture and agent design, and Person 4 writes UI, accessibility, and ethics.

**Interfaces between people:** Person 3 imports `generate()`, `check_quality()`, and `read_text()` from Persons 1 and 2. Agree on these function signatures in week 1 and use fake implementations until the real models land.

## 6. Timeline (6 weeks)

| Week | Person 1 | Person 2 | Person 3 | Person 4 |
|---|---|---|---|---|
| 1 | Download Flickr8k, build pipeline, pretrained BLIP baseline | Download VizWiz quality data, OpenCV baseline, EasyOCR wrapper | FastAPI skeleton with stub tools, API contract | Expo app that takes a photo, sends it, and speaks the response |
| 2 | First Flickr8k training run, scale to Flickr30k | Train quality classifier | Router and composer working with stubs, logging | Polish UI, large buttons, accessibility labels, connect to real server |
| 3 | Tune the captioner, add attention if time allows | Finish quality model, evaluate vs. baseline, collect own test photos | Integrate real quality and OCR models | Test on a real phone with a screen reader, fix UI issues |
| 4 | VizWiz fine-tune | Help build evaluation set, run OCR checks | Integrate captioner, end-to-end test | Recruit users, prepare test script |
| 5 | Final evaluation and failure analysis | Evaluate quality and OCR, agent-level action accuracy | Fix bugs, freeze demo setup, record backup demo video | User testing (5 to 10 people), collect ratings |
| 6 | Report sections, slides | Report sections, slides | Report sections, demo rehearsal | Compile report, slides, final rehearsal |

**Checkpoint at end of week 3:** the whole pipeline works from phone to spoken output, even if the captioner is still pretrained BLIP as a placeholder. Anything after that is improvement, not rescue.

## 7. Datasets

| Model | Dataset | Notes |
|---|---|---|
| Captioner (pretrain) | Flickr30k (or a COCO subset of 20,000 to 40,000 images) | Clean photos, good captions. Start with Flickr8k to debug the pipeline |
| Captioner (fine-tune) | VizWiz-Captions | Photos taken by blind users. Evaluate on the validation split because test labels are hidden |
| Quality classifier | VizWiz Image Quality Assessment | Plus synthetic blur, darkening, and cropping on good images |
| Evaluation | VizWiz validation subset + your own photos | 30 to 50 images with manual scoring |

Verify download links and licenses on the official pages, and cite each dataset in the report.

## 8. Evaluation

- **Quality classifier:** accuracy, precision, recall, versus OpenCV rules.
- **Captioner:** BLEU-4, CIDEr, or CLIPScore on a held-out set, versus pretrained BLIP.
- **OCR:** word-level correctness on text images.
- **Agent level:** how often the router picks the right action (retake, caption, OCR).
- **System:** latency per request.
- **Human:** 5 to 10 users rate usefulness and clarity on a short scale.
- **Training details:** loss curves, overfitting checks, train/validation/test split.

## 9. Definition of done

- Take a photo in the app and hear a description within a few seconds.
- A blurry or dark photo triggers a spoken retake request.
- "Read text" reads printed text aloud, with a warning when OCR confidence is low.
- The trained captioner and quality classifier are in use, with results against their baselines.
- Logs and results exist for at least 30 images.
- The app is tested with a screen reader and a handful of real users.
- The report covers PEAS, agent type, results, and limitations (hallucination, privacy, not for safety-critical use).

## 10. Cut list (drop in this order if behind)

1. Attention in the decoder.
2. The VizWiz fine-tune (report Flickr30k only).
3. The user study (use teammates instead).
4. The learned quality classifier (keep the OpenCV rules and report that honestly).

## 11. Risks to settle early

- **Instructor check:** confirm that EasyOCR and pretrained backbones count as acceptable.
- **Network:** the phone and server must be on the same Wi-Fi, or you need a tunnel. Test this in week 1, not on demo day.
- **GPU:** time the ROCm install on the RX 9070 XT. If it takes more than half a day, switch to Kaggle.
- **Free GPU limits:** save checkpoints every epoch and cache encoder features.
- **Demo backup:** record a video of the working app.
- **Ethics:** hallucinated descriptions, privacy of photos (faces, documents), bias on local scenes, and a clear disclaimer that the app is not for navigation or medication decisions.
