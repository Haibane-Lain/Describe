# Build plan: remaining scope against the proposal objectives

Companion to [status-report.md](status-report.md). That document establishes what
is done and what is broken; this one maps the concept proposal's objectives onto
work that remains, with dependencies, effort estimates, and the decisions that
have to be made before the estimates mean anything.

Effort is in **ideal person-days** for one competent person already familiar with
the codebase. They are not calendar days: the project is a four-person team, but
almost all of the ML work is bound to a single GPU on a single machine, so
parallelism is available only across the app / backend / writing tracks.

---

## 1. Where the objectives stand

> **Captioner scope updated.** Objective 4.3 is now done: the deployed captioner
> is a fine-tuned BLIP at BLEU-4 0.1958 on the full validation split, with a
> zero-shot baseline at 0.1572 and the from-scratch model reported as the
> ablation. Procedure and all measured numbers in
> [../finetune/README.md](../finetune/README.md). Decision A below is therefore
> resolved in favour of A2, and the 6-9 day estimate was replaced by about one
> day of work once the cached-feature path was in place.

| § | Objective | State | Remaining |
|---|---|---|---|
| 4.1 | Mobile app, React Native + Expo, screen-reader operable | Not started (HTML mockup only) | 10–14 d |
| 4.2 | Image quality gate | **Complete** | 0.5 d (routing table re-check) |
| 4.3 | Captioning model + BLIP baseline + ablation | **Complete** — BLIP fine-tuned, baseline and ablation both measured | 0 |
| 4.4 | OCR module + preprocessing | **Built and measured** — `train/ocr/`, 11.4% OCR vs 11.7% captioner on the 577 reading questions, 20.3% combined. See [ocr-measurement.md](ocr-measurement.md) | 1–2 d (word-level set for proposal §5C is uncollected; the harness is ready) |
| 4.5 | Router + response composer | Not started | 3–4 d |
| 4.6 | Guided capture + best-frame selection | Not started | 3–4 d |
| 4.7 | End-to-end evaluation + user study | Not started | 6–10 d |
| 5A.2 | FastAPI `/describe` backend | Not started | 3–4 d |

**Total: roughly 31–46 ideal person-days**, with the captioner no longer on the
critical path. The captioner's live inference path
(`train/finetune/predict.py`, verified against the cached path and measuring
~750 ms/image) is ready for the backend to call.

---

## 2. The two decisions that gate everything

### Decision A — what generates the description

Until the token-sensitivity probe in §5 of the status report is run, the captioner
effort cannot be estimated honestly. Two branches:

| Branch | Condition | Consequence |
|---|---|---|
| **A1 — repair the from-scratch model** | Probe shows the next-token distribution does *not* respond to the previous token (token signal swamped) | 2–3 d to fix the forward pass and re-run; then the existing training code is reusable and objective 3 is fully met as written |
| **A2 — BLIP becomes the deployed captioner** | Probe shows the decoder *does* condition on tokens, or the fix does not lift 4-gram match | 4–6 d for the fine-tune plus evaluation; the from-scratch model becomes the reported ablation |

**A2 is the plan of record**, because it is the only branch with a known-good
outcome. BLIP fine-tuned on VizWiz-Captions is a real captioner; the proposal
already commits to training exactly this model as its like-for-like comparison
(§4.3, §5A), so promoting it to production is a re-labelling, not a scope change.
A1 is worth the 15-minute probe and, if it works, the 2–3 days — but it must not
hold up the BLIP path, which should start in parallel.

**What changes if A2 is chosen.** The app, the gate, the router, the OCR branch,
the `/describe` contract and the latency budget are all unaffected. Only two
things change: the captioner table in the report gains a "pretrained baseline
(deployed)" row above the from-scratch rows, and objective 3's ablation is
reported as the primary result rather than as an appendix.

### Decision B — is the user study in scope

Objective 4.7 requires a user study with blind or low-vision participants, and
§5A.4 requires the instrument as a deliverable. This is the highest-variance item
in the plan, and it is **not** an engineering task: it needs recruitment, an
ethics/consent path through the department, and a device handling procedure. It
should be started first, because its lead time is longer than any coding task
here, and the instrument can be written before the app is finished.

If the study cannot be run within the term, the honest fallback is a structured
expert review (faculty plus an accessibility practitioner) with blind-folded
task completion, reported explicitly as *not* a substitute for BLV participants.
Decide this now — it changes what the app must support (e.g. remote usability
sessions need logging the app does not yet have).

---

## 3. Work breakdown

### Track 1 — Product (app, backend, integration)

**1.1 FastAPI `/describe` backend — 3–4 d.** Implement the contract in proposal
§5A verbatim: `multipart` image + `mode`, returning `status`, `message`, `reason`,
per-tool `confidence`, `tools_used`, `latency_ms`. Two properties from the
proposal must be preserved because the app depends on them: confidence is reported
per tool (the values are not comparable, and the hedging rule needs separate
inputs), and `mode` is a hint rather than a command.

**1.2 Per-stage latency budget — 1 d.** The proposal commits to reporting latency
per stage on the demo hardware, not as one asserted figure. Instrument the three
stages now rather than retrofitting; §4.7 and the report both need it.

**1.3 Expo app — 10–14 d.** Largest single item. Sub-tasks, roughly ordered:

* Screen-reader operability and large high-contrast controls *first*, not last —
  accessibility that is retrofitted is always worse and usually incomplete.
* Guided capture: audio cues, burst/short-clip capture, repeat control.
* Automatic spoken playback of results via on-device TTS.
* The five UI states the mockups already define (`03-description`,
  `04-description-uncertain`, `05-read-text`, `06-read-text-unclear`,
  `07-retake-dark`, `08-retake-blurry`, `09-error-offline`) — the mockups in
  `prototype/figma/screens/` are a real head start; treat them as the UI spec.
* Offline/connection-failure state with a spoken error, per §8.

**1.4 Best-frame selection (objective 4.6) — 3–4 d.** On-device scoring of a
burst to pick one frame. Deliberately deferred until after 1.3, because it sits
behind the same capture path.

### Track 2 — ML (GPU-bound; serialize)

**2.1 Token-sensitivity probe — 0.25 d.** First thing run. Decides A1 vs A2.
Fix `train/diagnose.py`'s mask offset and verdict rule at the same time (status
report §4.3), so the diagnostic is trustworthy for everything after.

**2.2 BLIP fine-tune on VizWiz-Captions — 4–6 d.** Includes wiring a
`transformers`/LAVIS dependency, a fine-tune run, and evaluation on the same
VizWiz val split the existing from-scratch numbers use, so the comparison is
like-for-like. **Risk:** this is a new dependency stack on a ROCm machine with a
documented history of driver-level crashes under sustained load
(`docs/training-status.md`, `docs/environment.md`). Budget a day for that alone,
and run it at a conservative batch size rather than rediscovering the crash.

**2.3 From-scratch ablation, re-evaluated — 1 d.** If A1 succeeds, re-run the
existing pipeline with the fix. If A1 fails, nothing to do beyond what exists —
the current numbers already serve as the ablation, and the collapse analysis in
the status report is the explanation.

**2.4 OCR module + preprocessing (objective 4.4) — 5–7 d.** EasyOCR wrapped as an
internal tool, behind perspective correction, deskewing, and upscaling, with
explicit low-confidence handling. Two pieces of work beyond integration:

> **Status: the module is built and measured** — `train/ocr/`, measured in
> [ocr-measurement.md](ocr-measurement.md). What the measurement adds to the two
> notes below:
>
> * **Text-region routing.** The `text_detected` flag turns out to come from the
>   VizWiz *caption annotations* (`train/data/prepare.py`), so it is an offline
>   label, not a runtime signal — the router still needs a real text-presence
>   detector. The measurement says how much it is worth: on the 28 reading images
>   where crowdworkers reported no text, OCR scores **0.0%**, against 12.0% on
>   the 549 where they did. Invoking OCR on images with no text is pure cost.
> * **The evaluated text set is still uncollected.** `train/ocr/word_metrics.py`
>   provides CER, WER and exact match against
>   `data/raw/ocr_eval/ground_truth.jsonl`; the photographs themselves remain a
>   team task, and they are the only part of objective 4.4 still outstanding.
>
> The measurement also settles the configuration question: canvas 2560 costs 4.8×
> the time of 1600 for no measurable accuracy difference, so the frozen
> configuration in `train/configs/ocr_frozen.json` is the cheaper one. Geometry
> preprocessing changed no total (66/577 either way) and cost nothing measurable,
> so it is kept for the conditions the 577 happen not to contain.

* **Text-region routing.** The evaluation already reports a `text_detected` split
  and found the captioner *better* on text-bearing images
  (`runs/vizwiz_ft2/eval_vizwiz_val.json`). The router (2.5) needs a cheap
  text-presence signal to decide whether to invoke OCR at all. Reuse whatever the
  captioner's `text_detected` flag was reconstructed from, rather than building a
  new detector.
* **A small evaluated text set.** Proposal §5C requires word-level correctness on
  text images, and explicitly lists "collected label and document
  photographs" — meaning the team must photograph labels and documents. This
  cannot be done from existing datasets and needs to be scheduled early enough
  that a decent spread of conditions is captured.

**2.5 Router + response composer (objective 4.5) — 3–4 d.** Rule-based selection
of retake / caption / OCR, then conversion of tool outputs into short spoken
sentences that lead with warnings and qualify uncertain descriptions. The
asymmetric-cost rule from §5B is the core: bound the false-accept rate, accept
unnecessary retakes.

Note the gate's limits here: per-flaw AP for `obstructed` (0.203), `too bright`
(0.383) and `rotated` (0.336) is weak, and `other` has 9 positives. The
condition-to-action table in §5B should route on **unrecognizability** (which is
well measured, AUC 0.964) and treat the fine-grained flaw label as advisory. A
router that trusts the flaw head for `rotated` — the one condition where the
proposal promises automatic correction rather than a retake — will misroute.

### Track 3 — Evaluation and write-up

**3.1 Held-out end-to-end evaluation (objective 4.7) — 2–3 d.** Per-image logs of
scores, tools used, output and latency; the metric tables; routing decision
accuracy with asymmetric cost weighting. The infrastructure for this mostly
exists in `train/train/evaluate.py`; the work is running it through the assembled
system rather than the single components.

**3.2 User study instrument + sessions — 4–7 d.** See Decision B. Start the
recruitment and consent path immediately; the instrument is ~1 d of writing.

**3.3 Report, including the negative result — 3–4 d.** The from-scratch captioner
result belongs in the report as a documented finding with its diagnosis, not
buried. It is the most intellectually honest thing in the project and it costs
nothing extra to report well.

---

## 4. Suggested sequencing

```
Week 1   probe (2.1) ──┬─ BLIP fine-tune (2.2) ──────────┐
                       │                                 │
         backend contract (1.1) ── latency (1.2)         │
                       │                                 │
         OCR set captured (2.4) ── OCR module (2.4)      │
                       │                                 │
         user-study recruitment starts (3.2) ────────────┤
                                                         │
Week 2   Expo app accessibility + capture (1.3)          │
         router + composer (2.5)  ───────────────────────┤
                                                         │
Week 3   app UI states + TTS (1.3) ── integration ───────┘
         best-frame selection (1.4)         │
                                           │
Week 4   end-to-end eval (3.1) ── user study (3.2) ── report (3.3)
```

Critical path: **token-sensitivity probe → BLIP fine-tune → router → end-to-end
evaluation → report.** The app and the OCR/eval-set work are the only substantial
items that can proceed in parallel with GPU work, which is why they should start
immediately rather than after the captioner settles.

---

## 5. Risks, ordered by expected damage

| Risk | Impact | Mitigation |
|---|---|---|
| BLIP fine-tune is blocked by the ROCm/driver instability documented in `docs/environment.md` | Kills the deployed-captioner plan | Run at the conservative configuration; if the GPU wedges, BLIP inference on CPU is slow but sufficient to *demonstrate* the system, and the fine-tune can move to a rented GPU. Decide early. |
| User study cannot be recruited in time | Objective 4.7 partially unmet | Start recruitment now; prepare the expert-review fallback and label it honestly. |
| Router trusts weak per-flaw heads | Wrong action for rotated/bright/obstructed images | Route on unrecognizability; treat flaw label as advisory. |
| App accessibility retrofitted late | Rework, and a deliverable that fails objective 4.1's core claim | Build screen-reader operability into the first app milestone. |
| Text-eval set captured too late | OCR has no honest evaluation, §5C unmet | Photograph labels/documents in week 1, before the OCR module is finished. |
| Continue the COCO run "just in case" | 45 h of GPU time that delays BLIP, on a hypothesis the evidence contradicts | Stop it. |
| Report overstates the captioner | Undermines the project's stated contribution | Report BLEU-4 as at/near zero, and report the collapse analysis. |

---

## 6. What to do first, in order

1. **Stop the COCO run** and prune the 11.97 GB of checkpoints, keeping the four
   evaluated ones plus `quality_gate`, `flickr30k_attn3`, `vizwiz_ft2` and
   `coco_e2e` if its trajectory is to be cited.
2. **Run the token-sensitivity probe** and fix `train/diagnose.py` (§4.3 of the
   status report).
3. **Start the BLIP fine-tune** in parallel with step 2's decision, since A2 is
   the plan of record.
4. **Start user-study recruitment** — the longest lead time in the project.
5. **Write the `/describe` contract and latency instrumentation** — the interface
   every other component must satisfy.
6. **Begin the Expo app with accessibility first.**
7. **Capture the text-image evaluation set** while the OCR module is being built.
