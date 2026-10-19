//camera
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'app_settings.dart';
import 'speech.dart';
import 'theme.dart';
import 'voice.dart';
import 'widgets.dart';


//------------------------------------------------------------------------------
// Keeps ONE camera controller for the whole app, so a new camera screen
// reuses it instead of opening (and closing) the camera again.
class _CameraHolder {
  static CameraController? _controller;

  static Future<CameraController> get() async {
    final existing = _controller;
    if (existing != null && existing.value.isInitialized) return existing;

    final cameras = await availableCameras();
    final controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false, // silent clip
    );
    await controller.initialize();
    _controller = controller;
    return controller;
  }
}


//------------------------------------------------------------------------------
// SCREEN 2 — Camera / 5-second video capture
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  static const _clipSeconds = 5;

  CameraController? _controller;
  String? _error;
  bool _recording = false;
  bool _pinched = false;
  int _secondsLeft = _clipSeconds;

  @override
  void initState() {
    super.initState();
    _initCamera();
    // Start listening only after the prompt has finished.
    Speaker.speak(
      'Camera is ready. Say "Record or Capture", or pinch the screen, to record '
      '$_clipSeconds seconds of video.',
    ).then((_) {
      if (mounted && !_recording) _listenForCommands();
    });
  }

  void _listenForCommands() {
    Voice.start({
      'record': _recordClip,
      'capture': _recordClip,
    });
  }

  Future<void> _initCamera() async {
    try {
      final controller = await _CameraHolder.get();
      if (!mounted) return;
      setState(() => _controller = controller);
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera unavailable');
      Speaker.speak('The camera is unavailable. Check camera permission.');
    }
  }

  @override
  void dispose() {
    Voice.stop();
    // The camera controller is shared, so it is NOT disposed here.
    super.dispose();
  }

  // Records a 5-second clip, then sends its file path to the processing screen.
  Future<void> _recordClip() async {
    final controller = _controller;
    if (_recording || controller == null || !controller.value.isInitialized) {
      return;
    }

    Voice.stop(); // free the microphone and ignore the app's own prompt
    setState(() {
      _recording = true;
      _secondsLeft = _clipSeconds;
    });

    try {
      Speaker.speak(
        'Recording for $_clipSeconds seconds. Hold the phone steady.',
      );
      await controller.startVideoRecording();

      for (var i = _clipSeconds; i > 0; i--) {
        if (!mounted) {
          // Screen was left mid-recording: stop so the camera is free again.
          try {
            await controller.stopVideoRecording();
          } catch (_) {}
          return;
        }
        setState(() => _secondsLeft = i);
        await Future.delayed(const Duration(seconds: 1));
      }

      final file = await controller.stopVideoRecording();
      if (!mounted) return;
      await Navigator.pushNamed(context, '/processing', arguments: file.path);
    } catch (e) {
      Speaker.speak('Recording failed. Please try again.').then((_) {
        if (mounted && !_recording) _listenForCommands();
      });
    } finally {
      if (mounted) setState(() => _recording = false);
    }
  }

  // Pinch gesture = start recording (when Gestures Mode is on).
  void _onScaleUpdate(ScaleUpdateDetails d) {
    if (!AppSettings.instance.gestures || _pinched) return;
    if (d.scale < 0.8) {
      _pinched = true;
      _recordClip();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: SafeArea(
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onScaleStart: (_) => _pinched = false,
          onScaleUpdate: _onScaleUpdate,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                _buildHeader(),
                const SizedBox(height: 12),
                Expanded(child: _buildViewfinder()),
                const SizedBox(height: 12),
                BigButton(
                  color: _recording ? kRed : kYellow,
                  icon: _recording
                      ? Icons.fiber_manual_record
                      : Icons.videocam_outlined,
                  label: _recording
                      ? 'Recording... $_secondsLeft'
                      : 'Record $_clipSeconds seconds',
                  hint: _recording
                      ? 'Hold the phone steady'
                      : 'Speaks what is in front of you',
                  onTap: _recordClip,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Row(
      children: [
        const AppLogo(),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Describe',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              Text(
                _recording ? 'Recording...' : 'Camera is ready',
                style: const TextStyle(color: kMuted, fontSize: 11),
              ),
            ],
          ),
        ),
        Semantics(
          button: true,
          label: 'Help and settings',
          child: OutlinedButton.icon(
            onPressed: _recording
                ? null
                : () => Navigator.pushNamed(context, '/settings'),
            icon: const Icon(Icons.help_outline, size: 16),
            label: const Text('Help'),
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.white,
              side: BorderSide(color: borderColor()),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildViewfinder() {
    final controller = _controller;

    return Stack(
      fit: StackFit.expand,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(10),
          child: controller != null && controller.value.isInitialized
              ? FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: controller.value.previewSize!.height,
                    height: controller.value.previewSize!.width,
                    child: CameraPreview(controller),
                  ),
                )
              : Container(
                  color: kCard,
                  alignment: Alignment.center,
                  child: Text(
                    _error ?? 'Starting camera...',
                    style: const TextStyle(color: kMuted),
                  ),
                ),
        ),
        CustomPaint(painter: _BracketPainter()),
        if (_recording)
          Positioned(
            top: 16,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                color: Colors.black54,
                child: StatusBadge('REC  $_secondsLeft', color: kRed),
              ),
            ),
          ),
      ],
    );
  }
}


//------------------------------------------------------------------------------
// White corner brackets drawn over the preview.
class _BracketPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final p = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    const inset = 12.0;
    const len = 22.0;
    final l = inset, t = inset, r = size.width - inset, b = size.height - inset;

    canvas.drawPath(
      Path()
        ..moveTo(l, t + len)..lineTo(l, t)..lineTo(l + len, t)
        ..moveTo(r - len, t)..lineTo(r, t)..lineTo(r, t + len)
        ..moveTo(l, b - len)..lineTo(l, b)..lineTo(l + len, b)
        ..moveTo(r - len, b)..lineTo(r, b)..lineTo(r, b - len),
      p,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}