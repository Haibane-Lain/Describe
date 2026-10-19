import 'package:flutter_tts/flutter_tts.dart';
import 'app_settings.dart';


//------------------------------------------------------------------------------
// Text-to-speech wrapper. Every screen reads its content through this.
class Speaker {
  static final FlutterTts _tts = FlutterTts();
  static bool _configured = false;
  static int _count = 0;

  /// True while the app is talking, so voice commands can ignore it.
  static bool get speaking => _count > 0;

  /// Completes when the speech has finished.
  /// [force] speaks even when "Speak results aloud" is off (e.g. Repeat button).
  static Future<void> speak(String text, {bool force = false}) async {
    final s = AppSettings.instance;
    if (!s.speakResults && !force) return;

    if (!_configured) {
      await _tts.awaitSpeakCompletion(true);
      _configured = true;
    }
    await _tts.setSpeechRate(s.speechRate);
    await _tts.setLanguage('en-US');
    await _tts.stop();

    _count++;
    try {
      await _tts.speak(text);
    } finally {
      _count--;
    }
  }

  static Future<void> stop() => _tts.stop();
}