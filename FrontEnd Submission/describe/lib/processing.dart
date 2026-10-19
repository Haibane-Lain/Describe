//processing
import 'package:flutter/material.dart';
import 'api.dart';
import 'app_settings.dart';
import 'speech.dart';
import 'theme.dart';


//------------------------------------------------------------------------------
class ProcessingScreen extends StatefulWidget {
  const ProcessingScreen({super.key});

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  bool _started = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_started) return;
    _started = true;
    _run(ModalRoute.of(context)!.settings.arguments as String);
  }

  Future<void> _run(String videoPath) async {
    // Announce before the wait. For a blind user, a silent spinner is worse
    // than knowing something is happening.
    await Speaker.speak('Describing. Checking image quality.');
    if (!mounted) return;

    // Check the backend first. Without this, a dead server surfaces as a
    // confusing multipart failure after the user has already recorded; here it
    // becomes a clear spoken message naming the address to check.
    if (!await DescribeApi.isReachable()) {
      await Speaker.speak(
        'I cannot reach the describer at ${AppSettings.instance.backendUrl}. '
        'Please check the server is running.',
        force: true,
      );
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/camera', (route) => false);
      return;
    }
    if (!mounted) return;

    Map<String, dynamic> result;
    try {
      result = await DescribeApi.describe(videoPath);
    } on DescribeApiException catch (e) {
      // Never leave the user on a spinner. Speak the reason and go back.
      // `force: true` because a failure must be announced even when the
      // "Speak results aloud" setting is off - otherwise the app would return
      // to the camera in silence and the user would not know why.
      await Speaker.speak(e.message, force: true);
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/camera', (route) => false);
      return;
    } catch (e) {
      await Speaker.speak('Something went wrong. Please try again.', force: true);
      if (!mounted) return;
      Navigator.pushNamedAndRemoveUntil(context, '/camera', (route) => false);
      return;
    }

    if (!mounted) return;
    Navigator.pushReplacementNamed(context, '/result', arguments: result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: SafeArea(
        child: Center(
          child: Semantics(
            liveRegion: true,
            label: 'Describing. Checking image quality.',
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: const [
                SizedBox(
                  width: 80,
                  height: 80,
                  child: CircularProgressIndicator(
                    color: kYellow,
                    strokeWidth: 8,
                  ),
                ),
                SizedBox(height: 24),
                Text(
                  'Describing...',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                SizedBox(height: 6),
                Text(
                  'Checking image quality...',
                  style: TextStyle(color: kMuted, fontSize: 12),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}