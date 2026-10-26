# Application development

This branch (`app-dev`) is for working on the **Flutter app only**. The backend
and model are stable; changes here should not require any Python edits.

If you also need to run the backend on your own machine, read
[onboarding-nvidia.md](onboarding-nvidia.md) first — this document assumes the
backend is already reachable.

---

## 1. Get the branch

```bash
git lfs install
git clone -b app-dev https://github.com/Haibane-Lain/Describe.git
cd Describe
```

Already have a clone?

```bash
git fetch origin
git checkout app-dev
```

The app lives in **`FrontEnd Submission/describe/`**.

> The weights (~640 MB) come down over Git LFS. You **do not need them** to work
> on the app — but `git lfs install` must be present before cloning or the clone
> still succeeds, just with pointer files. If you are only touching the app you
> can skip the large files:
> ```bash
> GIT_LFS_SKIP_SMUDGE=1 git clone -b app-dev https://github.com/Haibane-Lain/Describe.git
> ```

---

## 2. Prerequisites

| | Version | Why |
|---|---|---|
| Flutter | **3.47.7** (stable) | the version this was built and verified with |
| Android SDK | **36** | Flutter 3.47 requires platform 36 |
| JDK | **17** | **21+ breaks the build** (see below) |
| Device | Android 7.0+ | `minSdk 24` |

### JDK 17 is not optional

With Java 21+ the Gradle build fails with:

```
Execution failed for task ':app:compileFlutterBuildDebug'.
> Cannot invoke "java.io.File.exists()" because "parent" is null
```

Install JDK 17 and point Flutter at it:

```powershell
flutter config --jdk-dir "C:\path\to\jdk-17"
```

### First-time setup

```powershell
cd "FrontEnd Submission\describe"
flutter pub get
flutter analyze
```

`flutter analyze` should report **2 info-level deprecations** in `voice.dart`
(`listenFor`, `pauseFor`). Those are pre-existing and harmless; see
[Known bugs](#5-known-bugs) if you want to clear them.

---

## 3. Run it

### On a device, with hot reload (the normal loop)

```powershell
flutter run -d <device-id>          # flutter devices lists ids
```

`flutter run` builds, installs, and attaches. Press `r` to hot-reload after an
edit, `R` for a full restart.

### Build an installable APK

```powershell
flutter build apk --release
# -> build/app/outputs/flutter-apk/app-release.apk  (~50 MB)
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Use `--release` when checking anything network- or permission-related: debug and
release builds have **different manifests** (see gotcha 2 below).

### You need a backend

The app is useless without it. On the machine running `train.serve`:

```powershell
python -m train.serve --serve --port 8000
```

Then in the app: **Settings → Describer address → Set address** →
`http://<backend-LAN-IP>:8000`.

Confirm from the phone's browser first: `http://<LAN-IP>:8000/health` must return
JSON. If it does not, the app cannot work and the problem is the network or
firewall, not your code.

---

## 4. App architecture

All under `FrontEnd Submission/describe/lib/`:

| File | Lines | Responsibility |
|---|---|---|
| `main.dart` | 37 | app entry + named routes |
| `splash.dart` | 53 | splash screen |
| `welcome.dart` | 91 | intro, speaks and listens for "Begin" |
| `camera.dart` | 270 | 5-second video capture, voice + pinch trigger |
| `processing.dart` | 103 | uploads the clip, awaits the verdict |
| `result.dart` | 226 | displays and speaks the description |
| `settings.dart` | 289 | settings screen |
| `app_settings.dart` | 27 | in-memory settings (see bug 2) |
| `api.dart` | 85 | **all HTTP** — the only file that knows the backend exists |
| `voice.dart` | 57 | speech-recognition command listener |
| `speech.dart` | 31 | text-to-speech wrapper |
| `widgets.dart` | 176 | shared buttons, badge |
| `theme.dart` | 10 | colours |

Read `api.dart` first. It is deliberately the single point of contact with the
backend, so the contract lives in one place.

### The flow

```
splash → welcome → camera ──(5 s clip)──▶ processing ──(JSON)──▶ result
                      ▲                                            │
                      └──────────── "retake" / "next" ──────────────┘
```

### The backend contract

Defined in **[train/docs/frontend-contract.md](../train/docs/frontend-contract.md)**.

```json
{
  "status": "ok" | "lowConfidence" | "tooDark",
  "text": "a person is holding a bottle of vitamin water.",
  "confidence": 100,
  "tags": ["basic caption"],
  "hint": ""
}
```

**Do not change this schema without agreeing it with the backend side.** The
Dart side parses it in `DescribeResult.fromMap` (`result.dart`).

---

## 5. Known bugs

These were found while reviewing the code, not from user testing. They are the
suggested starting point.

### 1. `confidence` is displayed as if it were real — **highest priority**

`result.dart:200` renders `'Confidence ${r.confidence}%'` with a progress bar at
`:207`.

But `confidence` is a **placeholder** from the backend: it is `100` for `ok` and
`0` otherwise. Calibrated confidence was never implemented. So the app currently
tells a blind user **"Confidence 100%"** on every successful description, which
is a fabricated number presented as a measurement.

For a tool aimed at blind users this is the worst class of bug: confidently wrong
information. Suggested options:

- Remove the confidence row and progress bar entirely, or
- Show the qualitative `status` only (`ok` / `low confidence` / `too dark`),
  which is real, or
- Keep the bar but label it "not available" until a real signal exists.

### 2. Settings do not persist — breaks the demo after a restart

`app_settings.dart` is pure in-memory state. There is no `shared_preferences`
dependency. **The backend address resets to `http://10.0.2.2:8000` every time the
app restarts**, which is the emulator-only address and fails on a real phone.

So: set the address, test, close the app, reopen — it is broken again and looks
like a network fault.

Fix: add `shared_preferences`, load in `AppSettings`, save on `update()`.

### 3. "Retake" and "Next" do the same thing

In `result.dart`, both `_backToCamera` and `_next` call
`Navigator.pushNamedAndRemoveUntil(context, '/camera', ...)`. They are identical.

Either make them differ (e.g. "next" should keep a session/history, or clear the
clip) or collapse them to a single action. Right now having two voice commands
for one behaviour is confusing for a user who cannot see the screen.

### 4. Voice commands are unverified on a device

`voice.dart` listens with `listenFor: 30 s` and `pauseFor: 4 s`, and only starts
*after* the description finishes. The commands `retake` / `next` / `repeat` have
**never been exercised on hardware** — they were written but not tested.

Watch what the recogniser hears:

```powershell
adb logcat | Select-String "flutter"
```

If words are missed, the fix is usually the pause window or adding synonyms in
the `Voice.start({...})` map in `result.dart`.

### 5. Pre-existing analyzer deprecations

`voice.dart:46,47` use `listenFor` / `pauseFor`, deprecated in
`speech_to_text 7.x` in favour of `SpeechListenOptions`. Fixing them changes
recogniser behaviour, so do it deliberately and re-test voice afterwards.

### 6. Minor

- `processing.dart` shows a spinner with no way to cancel, and no progress
  indication beyond the initial "Describing" announcement.
- `camera.dart` surfaces failures as a spoken "Recording failed" with no detail.
- `settings.dart` has no way to reset to defaults.

---

## 6. Test checklist

Work through this before calling the app done. Rows marked **unverified** have
never been run on hardware.

| # | Case | Expected | Status |
|---|---|---|---|
| 1 | Fresh install, launch | splash → welcome, speaks the intro | verified once |
| 2 | Say "Begin" | goes to camera | **unverified** |
| 3 | Say "Record or Capture" | records 5 s | **unverified** |
| 4 | Pinch the screen | records 5 s | **unverified** |
| 5 | Normal photo | speaks description + the retake/next options | **unverified** |
| 6 | Say "retake" | back to camera | **unverified** |
| 7 | Say "next" | back to camera | **unverified** |
| 8 | Say "repeat" | re-speaks the description | **unverified** |
| 9 | Dark room | `tooDark` + "Too dark…" hint spoken | **unverified** |
| 10 | Move while recording | `lowConfidence` + blur hint spoken | **unverified** |
| 11 | **Stop the backend, record** | speaks "I cannot reach the describer…" and returns to camera — **must not hang** | **unverified** |
| 12 | Restart the app after setting the address | address should survive (currently: bug 2) | fails today |
| 13 | Turn "Speak results aloud" off | errors still announced (they use `force: true`) | **unverified** |
| 14 | Rotate the phone / background and resume | no crash, camera recovers | **unverified** |

Test 11 matters most: a silent spinner in front of an audience is the worst
failure mode.

---

## 7. Gotchas

**1. `android/local.properties` is machine-specific.** It holds absolute paths to
the Flutter SDK and Android SDK, and is generated by the tooling. **Do not commit
it.** If a build fails complaining about a path you do not have, delete it and
rebuild:

```powershell
Remove-Item android\local.properties
flutter pub get
```

**2. `INTERNET` and cleartext traffic are declared in the manifest — keep them.**

`android/app/src/main/AndroidManifest.xml` contains:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<application ... android:usesCleartextTraffic="true">
```

Debug builds get `INTERNET` automatically from Flutter's debug manifest, so if
these go missing **debug still works and release fails** — the worst kind of
regression. The describer is reached over plain `http://` on a LAN address,
which Android 9+ blocks without the cleartext flag.

**3. The app package id is `com.example.describe_app`.** Renaming it changes
`adb uninstall` commands and the install-over behaviour; do it in one deliberate
commit if you do it at all.

**4. Release is signed with the debug key** (`android/app/build.gradle.kts` has a
TODO). Fine for sideloading and internal testing; **not publishable** to the Play
Store.

**5. `flutter analyze` is the fastest sanity check.** It caught a real defect in
the integration work (`unintended_html_in_doc_comment`). Run it before every
commit.

---

## 8. What not to change

| Don't | Why |
|---|---|
| The JSON contract in `DescribeResult.fromMap` | agreed with the backend; breaking it silently breaks descriptions |
| `api.dart`'s field name `video` in the multipart upload | the endpoint expects exactly that field |
| The three `DescribeStatus` enum names | they must match the backend's strings exactly (`ok`, `lowConfidence`, `tooDark`) |
| Add OCR/text-reading expectations | OCR is **disabled**; `tags` will not contain `"text read"` |

If you need a contract change, raise it before implementing — both sides have to
move together.

---

## 9. Reporting

When you have findings, report per bug:

- what you did
- what you expected
- what happened (and the `adb logcat` line if relevant)
- device model and Android version

That is enough to reproduce. For anything network-related, also say whether
`http://<LAN-IP>:8000/health` worked **from the phone's browser** — it separates
app bugs from environment problems immediately.
