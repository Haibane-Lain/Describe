import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';


//------------------------------------------------------------------------------
// Settings shared by every screen.
//
// These are persisted to disk. Before this existed the state was in-memory only,
// so the backend address reverted to the emulator default (`10.0.2.2`) on every
// app restart - which fails on a real phone and looks like a network fault.
//
// Call `await AppSettings.load()` once before `runApp`, so the first widget tree
// is built with the stored values rather than the defaults.
class AppSettings extends ChangeNotifier {
  AppSettings._();
  static final AppSettings instance = AppSettings._();

  static const _kSpeakResults = 'speakResults';
  static const _kGestures = 'gestures';
  static const _kVoice = 'voice';
  static const _kHighContrast = 'highContrast';
  static const _kSpeechRate = 'speechRate';
  static const _kLanguage = 'language';
  static const _kBackendUrl = 'backendUrl';

  /// The Android emulator's alias for the host machine. Correct on an emulator,
  /// wrong on a physical phone - see [backendUrl].
  static const defaultBackendUrl = 'http://10.0.2.2:8000';

  SharedPreferences? _prefs;

  bool speakResults = true;
  bool gestures = true;
  bool voice = true;
  bool highContrast = false;
  double speechRate = 0.5; // 0.0 (slow) to 1.0 (fast)
  String language = 'English (US)';

  /// Where the describe backend lives (train/serve.py).
  ///
  /// A phone cannot reach `localhost` - that resolves to the phone itself - so
  /// this is a setting rather than a constant.
  ///
  ///   Android emulator : http://10.0.2.2:8000
  ///   iOS simulator    : http://127.0.0.1:8000
  ///   Physical phone   : `http://YOUR-PC-LAN-IP:8000`
  String backendUrl = defaultBackendUrl;

  /// Reads stored settings into the singleton. Call once at startup.
  ///
  /// Every read falls back to the default, so a first run, a cleared store, or a
  /// key removed in a later version all behave sensibly.
  static Future<AppSettings> load() async {
    final settings = instance;
    try {
      final prefs = await SharedPreferences.getInstance();
      settings._prefs = prefs;
      settings.speakResults = prefs.getBool(_kSpeakResults) ?? settings.speakResults;
      settings.gestures = prefs.getBool(_kGestures) ?? settings.gestures;
      settings.voice = prefs.getBool(_kVoice) ?? settings.voice;
      settings.highContrast = prefs.getBool(_kHighContrast) ?? settings.highContrast;
      settings.speechRate = prefs.getDouble(_kSpeechRate) ?? settings.speechRate;
      settings.language = prefs.getString(_kLanguage) ?? settings.language;
      settings.backendUrl = prefs.getString(_kBackendUrl) ?? settings.backendUrl;
    } catch (_) {
      // A failed load must not stop the app launching: defaults are fine.
    }
    return settings;
  }

  /// Applies a change, persists it, and notifies listeners.
  ///
  /// The callback signature is unchanged, so every existing call site keeps
  /// working.
  void update(VoidCallback change) {
    change();
    _persist();
    notifyListeners();
  }

  void _persist() {
    final prefs = _prefs;
    if (prefs == null) return;
    // Fire-and-forget: the in-memory value is already correct, and a failed
    // write must never crash a settings toggle.
    prefs.setBool(_kSpeakResults, speakResults);
    prefs.setBool(_kGestures, gestures);
    prefs.setBool(_kVoice, voice);
    prefs.setBool(_kHighContrast, highContrast);
    prefs.setDouble(_kSpeechRate, speechRate);
    prefs.setString(_kLanguage, language);
    prefs.setString(_kBackendUrl, backendUrl);
  }
}
