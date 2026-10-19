//api
//
// Talks to the describe backend (train/serve.py). The contract is documented in
// train/docs/frontend-contract.md; the response map is fed straight into
// DescribeResult.fromMap, so this file is the only place that knows about HTTP.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import 'app_settings.dart';


//------------------------------------------------------------------------------
class DescribeApi {
  /// Uploads the recorded clip and returns the raw response map.
  ///
  /// Throws [DescribeApiException] with a message that is safe to speak aloud.
  /// Nothing here ever hangs: a timeout is treated as a failure so the app can
  /// tell the user instead of spinning forever.
  static Future<Map<String, dynamic>> describe(String videoPath) async {
    final base = AppSettings.instance.backendUrl;
    final uri = Uri.parse('$base/describe');
    final file = File(videoPath);

    if (!await file.exists()) {
      throw const DescribeApiException(
        'I could not find the recording. Please try again.',
      );
    }

    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('video', videoPath));

    try {
      final streamed = await request.send().timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode != 200) {
        // The backend returns {"detail": "..."} on failure.
        String detail = 'status ${response.statusCode}';
        try {
          final decoded = jsonDecode(response.body);
          if (decoded is Map && decoded['detail'] != null) {
            detail = decoded['detail'].toString();
          }
        } catch (_) {}
        throw DescribeApiException('The describer failed. $detail');
      }

      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) {
        throw const DescribeApiException(
          'The describer sent something I did not understand.',
        );
      }
      return decoded;
    } on TimeoutException {
      throw const DescribeApiException(
        'The describer took too long to answer. Please try again.',
      );
    } on SocketException {
      throw DescribeApiException(
        'I cannot reach the describer at $base. '
        'Please check the server is running.',
      );
    } on http.ClientException {
      throw DescribeApiException(
        'I cannot reach the describer at $base. '
        'Please check the server is running.',
      );
    }
  }

  /// True when the backend answers /health. Used to fail fast and politely.
  static Future<bool> isReachable() async {
    final base = AppSettings.instance.backendUrl;
    try {
      final response = await http
          .get(Uri.parse('$base/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}


//------------------------------------------------------------------------------
class DescribeApiException implements Exception {
  const DescribeApiException(this.message);
  final String message;

  @override
  String toString() => message;
}
