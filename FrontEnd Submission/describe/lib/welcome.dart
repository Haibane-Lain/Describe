//welcome
import 'package:flutter/material.dart';
import 'speech.dart';
import 'theme.dart';
import 'voice.dart';
import 'widgets.dart';


//------------------------------------------------------------------------------
// SCREEN 1 — Welcome
class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen> {
  static const message =
      'Welcome to Describe, an image recognition tool for the visually '
      'impaired. Voice activation and gesture input are enabled by default. '
      'To use the app, simply say "Record or Capture", or pinch the screen. '
      'To proceed, simply say "Begin".';

  @override
  void initState() {
    super.initState();
    // Start listening only after the welcome message has finished.
    Speaker.speak(message).then((_) {
      if (mounted) {
        Voice.start({
          'begin': _proceed,
          'proceed': _proceed,
          'start': _proceed,
        });
      }
    });
  }

  @override
  void dispose() {
    Voice.stop();
    super.dispose();
  }

  void _proceed() => Navigator.pushReplacementNamed(context, '/camera');

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const AppLogo(),
              const SizedBox(height: 16),
              Expanded(
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: kCard,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: borderColor()),
                  ),
                  child: const SingleChildScrollView(
                    child: Text(
                      'Welcome to Describe, an image recognition tool for the '
                      'visually impaired.\n\n'
                      'Voice activation and gesture input are enabled by '
                      'default. To use the app, simply say "Record or Capture", '
                      'or pinch the screen.\n\n'
                      'To proceed, simply say "Begin"',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        height: 1.4,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              BigButton(
                color: kYellow,
                icon: Icons.remove_red_eye_outlined,
                label: 'Proceed',
                onTap: _proceed,
              ),
            ],
          ),
        ),
      ),
    );
  }
}