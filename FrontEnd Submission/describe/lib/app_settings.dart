import 'package:flutter/material.dart';


//------------------------------------------------------------------------------
// Settings shared by every screen. Persist with shared_preferences later.
class AppSettings extends ChangeNotifier {
  AppSettings._();
  static final AppSettings instance = AppSettings._();

  bool speakResults = true;
  bool gestures = true;
  bool voice = true;
  bool highContrast = false;
  double speechRate = 0.5; // 0.0 (slow) to 1.0 (fast)
  String language = 'English (US)';

  /// Where the describe backend lives (train/serve.py).
  ///
  /// A phone cannot reach `localhost` - that resolves to the phone itself - so
  /// this is a setting rather than a constant. Defaults to the Android
  /// emulator's alias for the host machine.
  ///
  ///   Android emulator : http://10.0.2.2:8000
  ///   iOS simulator    : http://127.0.0.1:8000
  ///   Physical phone   : `http://YOUR-PC-LAN-IP:8000`
  String backendUrl = 'http://10.0.2.2:8000';

  void update(VoidCallback change) {
    change();
    notifyListeners();
  }
}
