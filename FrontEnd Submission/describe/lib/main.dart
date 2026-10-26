import 'package:flutter/material.dart';
import 'app_settings.dart';
import 'camera.dart';
import 'processing.dart';
import 'result.dart';
import 'settings.dart';
import 'splash.dart';
import 'theme.dart';
import 'welcome.dart';


Future<void> main() async {
  // SharedPreferences needs the binding up before it is used.
  WidgetsFlutterBinding.ensureInitialized();

  // Load persisted settings BEFORE the first frame. Screens read
  // AppSettings.instance during build (camera.dart reads `gestures`, api.dart
  // reads `backendUrl`), so building first would show - and act on - defaults.
  await AppSettings.load();

  runApp(const App());
}


class App extends StatelessWidget {
  const App({super.key});


  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: AppSettings.instance,
      builder: (context, _) {
        final contrast = AppSettings.instance.highContrast;

        return MaterialApp(
          title: 'Describe',
          debugShowCheckedModeBanner: false,
          theme: ThemeData.dark().copyWith(
            scaffoldBackgroundColor: contrast ? Colors.black : kBg,
          ),
          home: const SplashScreen(),
          routes: {
            '/welcome': (_) => const WelcomeScreen(),
            '/camera': (_) => const CameraScreen(),
            '/processing': (_) => const ProcessingScreen(),
            '/result': (_) => const ResultScreen(),
            '/settings': (_) => const SettingsScreen(),
          },
        );
      },
    );
  }
}
