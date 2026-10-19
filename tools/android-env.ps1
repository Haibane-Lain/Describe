# Puts the Android toolchain on PATH for the current shell.
#
# Nothing is installed system-wide, because this machine has no admin rights and
# `C:\` root is not writable. Everything lives in user space:
#
#   Flutter    C:\Users\<you>\flutter
#   Android    C:\Users\<you>\AppData\Local\Android
#   JDK 17     C:\Users\<you>\AppData\Local\jdk17\jdk-17.0.13+11
#
# Java 26 is also installed and Flutter's Android build FAILS under it, so
# `flutter config --jdk-dir` has been pointed at the JDK 17 above. That setting
# is stored in Flutter's config, not here.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   . .\tools\android-env.ps1
#
# The first line is needed because this machine's CurrentUser execution policy is
# `RemoteSigned`, which refuses to run an unsigned local .ps1. `-Scope Process`
# affects only the current shell and reverts when it closes - it does not weaken
# the machine's policy. Alternatively, run a whole command with:
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ". .\tools\android-env.ps1; adb devices"

$flutterBin  = "$env:USERPROFILE\flutter\bin"
$platformTools = "$env:LOCALAPPDATA\Android\platform-tools"
$cmdlineTools = "$env:LOCALAPPDATA\Android\cmdline-tools\latest\bin"
$jdk17 = "$env:LOCALAPPDATA\jdk17\jdk-17.0.13+11"

$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:JAVA_HOME = $jdk17

$env:PATH = "$flutterBin;$platformTools;$cmdlineTools;$jdk17\bin;$env:PATH"

Write-Host "Android toolchain ready" -ForegroundColor Green
Write-Host "  flutter  : $flutterBin"
Write-Host "  adb      : $platformTools"
Write-Host "  ANDROID_HOME : $env:ANDROID_HOME"
Write-Host "  JAVA_HOME    : $env:JAVA_HOME"
Write-Host ""
Write-Host "The describe app project:"
Write-Host '  cd "C:\Coding\General Image Recognizer\frontend submission\describe"'
