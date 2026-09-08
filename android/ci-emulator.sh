#!/usr/bin/env bash
# Uses an already licensed SDK; never auto-accepts Android terms.
set -euo pipefail
test -s "$ANDROID_HOME/licenses/android-sdk-license" || { echo "Operator SDK license review required"; exit 1; }
sdkmanager "platforms;android-36" "build-tools;35.0.0" "emulator" "system-images;android-35;default;x86_64"
export ANDROID_AVD_HOME="${RUNNER_TEMP:-/tmp}/face-kit-avd"
mkdir -p "$ANDROID_AVD_HOME"
echo no | avdmanager create avd -n face-kit-test -k "system-images;android-35;default;x86_64" --force
"$ANDROID_HOME/emulator/emulator" -avd face-kit-test -no-window -no-audio -no-snapshot -gpu swiftshader_indirect -no-boot-anim >/tmp/face-kit-emulator.log 2>&1 &
emulator_pid=$!
trap 'adb emu kill >/dev/null 2>&1 || true; kill "$emulator_pid" 2>/dev/null || true' EXIT
timeout 90 adb wait-for-device || { tail -80 /tmp/face-kit-emulator.log; exit 1; }
booted=false
for attempt in $(seq 1 180); do
  if [[ "$(adb shell getprop sys.boot_completed | tr -d '\r')" == "1" ]]; then booted=true; break; fi
  sleep 2
done
$booted || { echo "Emulator did not boot"; tail -80 /tmp/face-kit-emulator.log; exit 1; }
adb shell settings put global window_animation_scale 0
adb shell settings put global transition_animation_scale 0
adb shell settings put global animator_duration_scale 0
"$@"
