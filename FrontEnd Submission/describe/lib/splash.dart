//splash
import 'package:flutter/material.dart';
import 'speech.dart';
import 'theme.dart';
import 'widgets.dart';


//------------------------------------------------------------------------------
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Speaker.speak('Describe. Image description and text reading.');
    Future.delayed(const Duration(seconds: 5), () {
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/welcome');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: const [
            AppLogo(size: 88),
            SizedBox(height: 16),
            Text(
              'Describe',
              style: TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 4),
            Text(
              'Image description & text reading',
              style: TextStyle(color: kMuted, fontSize: 13),
            ),
            SizedBox(height: 32),
            CircularProgressIndicator(color: kYellow, strokeWidth: 3),
          ],
        ),
      ),
    );
  }
}
