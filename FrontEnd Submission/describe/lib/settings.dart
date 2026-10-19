//settings
import 'package:flutter/material.dart';
import 'app_settings.dart';
import 'theme.dart';


//------------------------------------------------------------------------------
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  static const _languages = ['English (US)', 'English (UK)', 'Filipino'];

  @override
  Widget build(BuildContext context) {
    final s = AppSettings.instance;

    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        return Scaffold(
          backgroundColor: kBg,
          body: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: kCard,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: borderColor()),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildTitleRow(context),
                    const SizedBox(height: 8),
                    _switchRow(
                      'Speak results aloud',
                      'Uses the phone\'s voice to read results.',
                      s.speakResults,
                      (v) => s.update(() => s.speakResults = v),
                    ),
                    _switchRow(
                      'Gestures Mode',
                      'Use gestures to navigate the app.',
                      s.gestures,
                      (v) => s.update(() => s.gestures = v),
                    ),
                    _switchRow(
                      'Voice Mode',
                      'Use voice to navigate the app.',
                      s.voice,
                      (v) => s.update(() => s.voice = v),
                    ),
                    _buildSpeechRate(s),
                    _switchRow(
                      'Extra-high contrast',
                      'Pure black background and brighter outlines.',
                      s.highContrast,
                      (v) => s.update(() => s.highContrast = v),
                    ),
                    _buildLanguage(s),
                    _buildBackendUrl(context, s),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildTitleRow(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text(
                'Help & settings',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                'Adjust how this app speaks and listens.',
                style: TextStyle(color: kMuted, fontSize: 11),
              ),
            ],
          ),
        ),
        OutlinedButton.icon(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.close, size: 16),
          label: const Text('Done'),
          style: OutlinedButton.styleFrom(
            foregroundColor: Colors.white,
            backgroundColor: kBg,
            side: BorderSide(color: borderColor()),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
        ),
      ],
    );
  }

  Widget _switchRow(
    String title,
    String subtitle,
    bool value,
    ValueChanged<bool> onChanged,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: borderColor())),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                Text(
                  subtitle,
                  style: const TextStyle(color: kMuted, fontSize: 10),
                ),
              ],
            ),
          ),
          Switch(
            value: value,
            onChanged: onChanged,
            activeThumbColor: Colors.white,
            activeTrackColor: kGreen,
          ),
        ],
      ),
    );
  }

  /// Where the describe backend lives. Editable because a phone cannot reach
  /// `localhost`: an Android emulator needs 10.0.2.2, a physical phone needs the
  /// development machine's LAN address.
  Widget _buildBackendUrl(BuildContext context, AppSettings s) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: borderColor())),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Describer address',
            style: TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            s.backendUrl,
            style: const TextStyle(color: kMuted, fontSize: 10),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: () =>
                      s.update(() => s.backendUrl = 'http://10.0.2.2:8000'),
                  child: const Text('Emulator'),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton(
                  onPressed: () => _promptBackendUrl(context, s),
                  child: const Text('Set address'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _promptBackendUrl(BuildContext context, AppSettings s) async {
    final controller = TextEditingController(text: s.backendUrl);
    final value = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Describer address'),
        content: TextField(
          controller: controller,
          autofocus: true,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            hintText: 'http://192.168.1.10:8000',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text.trim()),
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (value != null && value.isNotEmpty) {
      s.update(() => s.backendUrl = value);
    }
  }

  Widget _buildSpeechRate(AppSettings s) {
    return Container(      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: borderColor())),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Speech rate',
            style: TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
          Row(
            children: [
              const Text('Slow', style: TextStyle(color: kMuted, fontSize: 10)),
              Expanded(
                child: Slider(
                  value: s.speechRate,
                  activeColor: kYellow,
                  onChanged: (v) => s.update(() => s.speechRate = v),
                ),
              ),
              const Text('Fast', style: TextStyle(color: kMuted, fontSize: 10)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLanguage(AppSettings s) {
    return Padding(
      padding: const EdgeInsets.only(top: 10),
      child: Row(
        children: [
          const Expanded(
            child: Text(
              'Language',
              style: TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          DropdownButton<String>(
            value: s.language,
            dropdownColor: kCard,
            style: const TextStyle(color: Colors.white, fontSize: 12),
            items: [
              for (final l in _languages)
                DropdownMenuItem(value: l, child: Text(l)),
            ],
            onChanged: (v) => s.update(() => s.language = v ?? s.language),
          ),
        ],
      ),
    );
  }
}
