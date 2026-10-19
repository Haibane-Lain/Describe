import 'package:flutter/foundation.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'app_settings.dart';
import 'speech.dart';


//------------------------------------------------------------------------------
// Listens for spoken commands. Pass phrases and what to do when each is heard.
class Voice {
  static final SpeechToText _stt = SpeechToText();
  static bool _ready = false;
  static bool _active = false;
  static Map<String, VoidCallback> _commands = {};

  static Future<void> start(Map<String, VoidCallback> commands) async {
    if (!AppSettings.instance.voice) return; // Voice Mode switch
    _commands = commands;
    _active = true;

    if (!_ready) {
      _ready = await _stt.initialize(
        onStatus: (status) {
          if (status == 'done' || status == 'notListening') _again();
        },
        onError: (_) => _again(),
      );
    }
    if (_ready && _active) _listen();
  }

  static void stop() {
    _active = false;
    _stt.cancel();
  }

  // Android stops listening after a pause, so start again.
  static void _again() {
    if (!_active) return;
    Future.delayed(const Duration(milliseconds: 600), () {
      if (_active && !_stt.isListening) _listen();
    });
  }

  static void _listen() {
    _stt.listen(
      listenFor: const Duration(seconds: 30),
      pauseFor: const Duration(seconds: 4),
      listenOptions: SpeechListenOptions(partialResults: true),
      onResult: (result) {
        // Ignore anything heard while the app itself is talking.
        if (!_active || Speaker.speaking) return;

        final heard = result.recognizedWords.toLowerCase();
        for (final entry in _commands.entries) {
          if (heard.contains(entry.key)) {
            final action = entry.value;
            stop();
            action();
            return;
          }
        }
      },
    );
  }
}