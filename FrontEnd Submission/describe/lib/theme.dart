import 'package:flutter/material.dart';
import 'app_settings.dart';


const Color kBg = Color(0xFF121212);
const Color kCard = Color(0xFF1E1E1E);
const Color kYellow = Color(0xFFFFD60A);
const Color kRed = Color(0xFFFF5C5C);
const Color kGreen = Color(0xFF2ECC71);
const Color kMuted = Color(0xFFB0B0B0);


Color borderColor() =>
    AppSettings.instance.highContrast ? Colors.white : const Color(0xFF3A3A3A);
