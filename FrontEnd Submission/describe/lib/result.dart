//result
import 'package:flutter/material.dart';
import 'speech.dart';
import 'theme.dart';
import 'voice.dart';
import 'widgets.dart';


//------------------------------------------------------------------------------
enum DescribeStatus { ok, lowConfidence, tooDark }


//------------------------------------------------------------------------------
class DescribeResult {
  const DescribeResult({
    required this.status,
    required this.text,
    this.tags = const [],
    this.hint = '',
  });

  final DescribeStatus status;
  final String text;
  final List<String> tags;
  final String hint;

  // There is deliberately no `confidence` field. The backend used to send a
  // placeholder (100 for ok, 0 otherwise) and this screen displayed it as
  // "Confidence 100%", which was a fabricated number shown to a user who cannot
  // check it. A real score was measured and rejected: sequence likelihood tracks
  // caption length more than caption quality, so the percentage would have been
  // meaningless. `status` is the honest signal and is what the UI uses.
  factory DescribeResult.fromMap(Map<String, dynamic> m) => DescribeResult(
        status: DescribeStatus.values.firstWhere(
          (s) => s.name == m['status'],
          orElse: () => DescribeStatus.ok,
        ),
        text: m['text'] as String? ?? '',
        tags: List<String>.from(m['tags'] ?? const []),
        hint: m['hint'] as String? ?? '',
      );
}


//------------------------------------------------------------------------------
class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  DescribeResult? _result;

  static const _lowConfidenceWarning =
      'I am not fully confident about this description. Treat it as a guess.';

  /// Spoken after the description, so the user knows what they can do next
  /// without seeing the screen. The phrases match the voice commands below.
  static const _optionsPrompt =
      'You can say "retake" to take another picture, or "next" to continue.';

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_result != null) return;
    _result = DescribeResult.fromMap(
      ModalRoute.of(context)!.settings.arguments as Map<String, dynamic>,
    );
    // Announce the result, then the choices, then start listening. Voice.start
    // must come after the speech: Voice ignores anything it hears while the app
    // is talking, so starting first would swallow the user's reply.
    Speaker.speak('${_spokenText(_result!)} $_optionsPrompt').then((_) {
      if (!mounted) return;
      _listen();
    });
  }

  void _listen() {
    Voice.start({
      // Retake: go back to the camera and record again.
      'retake': _backToCamera,
      're-take': _backToCamera,
      'retry': _backToCamera,
      'again': _backToCamera,
      'another': _backToCamera,
      // Next: accept this description and move on.
      'next': _next,
      'next picture': _next,
      'continue': _next,
      'done': _next,
      'repeat': () => Speaker.speak(_spokenText(_result!), force: true),
      'say it again': () => Speaker.speak(_spokenText(_result!), force: true),
    });
  }

  /// Accepts the current description and returns to the camera for the next one.
  void _next() => _backToCamera();

  @override
  void dispose() {
    Voice.stop();
    super.dispose();
  }

  String _spokenText(DescribeResult r) {
    switch (r.status) {
      case DescribeStatus.ok:
        return r.text;
      case DescribeStatus.lowConfidence:
        return '${r.text} $_lowConfidenceWarning';
      case DescribeStatus.tooDark:
        return '${r.text} ${r.hint}';
    }
  }

  void _backToCamera() => Navigator.pushNamedAndRemoveUntil(
        context,
        '/camera',
        (route) => false,
      );

  @override
  Widget build(BuildContext context) {
    final r = _result!;
    final dark = r.status == DescribeStatus.tooDark;
    final low = r.status == DescribeStatus.lowConfidence;
    final warn = dark || low;

    return Scaffold(
      backgroundColor: kBg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Align(
                alignment: Alignment.centerLeft,
                child: dark
                    ? const StatusBadge('RETAKE NEEDED', color: kRed)
                    : const StatusBadge('DESCRIPTION', color: kGreen),
              ),
              const SizedBox(height: 12),
              Text(
                dark ? "It's too dark" : 'Here is what I see',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              Semantics(
                liveRegion: true,
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: kCard,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                      color: warn ? kRed : borderColor(),
                      width: warn ? 1.5 : 1,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        r.text,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          height: 1.35,
                        ),
                      ),
                      if (dark) ...[
                        const SizedBox(height: 8),
                        Text(
                          r.hint,
                          style: const TextStyle(color: kMuted, fontSize: 13),
                        ),
                      ],
                      if (low) ...[
                        const SizedBox(height: 8),
                        const Text(
                          _lowConfidenceWarning,
                          style: TextStyle(color: kRed, fontSize: 12),
                        ),
                      ],
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(children: [for (final t in r.tags) TagChip(t)]),
              const Spacer(),
              if (dark) ...[
                BigButton(
                  color: kRed,
                  icon: Icons.camera_alt_outlined,
                  label: 'Retake picture',
                  hint: 'Say "retake"',
                  onTap: _backToCamera,
                ),
                const SizedBox(height: 10),
                GhostButton(label: 'Cancel', onTap: _backToCamera),
              ] else ...[
                BigButton(
                  color: kYellow,
                  icon: Icons.replay,
                  label: 'Repeat',
                  hint: 'Say "repeat"',
                  onTap: () => Speaker.speak(_spokenText(r), force: true),
                ),
                const SizedBox(height: 10),
                GhostButton(
                  label: 'Next picture',
                  hint: 'Say "next"',
                  onTap: _next,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}