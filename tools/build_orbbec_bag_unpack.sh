#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SDK_ROOT="${ORBBEC_SDK_ROOT:-$ROOT/OrbbecSDK-v1.10.22}"
OUTPUT="$ROOT/tools/bin/orbbec_bag_unpack"
SDK_LIB="$SDK_ROOT/lib/linux_x64"

# The SDK's bundled libob_usb returns LIBUSB_ERROR_OTHER in WSL instances
# without /sys/bus/usb, even for file-only playback. Its imported symbols use
# the standard libusb-1.0 ABI, whose system build initializes correctly there.
if grep -qi microsoft /proc/sys/kernel/osrelease && [[ ! -e /sys/bus/usb ]]; then
  RUNTIME="$ROOT/tools/orbbec_runtime_wsl"
  mkdir -p "$RUNTIME"
  cp -f "$SDK_LIB/libOrbbecSDK.so.1.10.22" "$RUNTIME/"
  cp -f "$SDK_LIB/liblive555.so" "$RUNTIME/"
  cp -f "$SDK_LIB/libdepthengine.so.2.0" "$RUNTIME/"
  ln -sfn libOrbbecSDK.so.1.10.22 "$RUNTIME/libOrbbecSDK.so.1.10"
  ln -sfn libOrbbecSDK.so.1.10 "$RUNTIME/libOrbbecSDK.so"
  ln -sfn libdepthengine.so.2.0 "$RUNTIME/libdepthengine.so"
  ln -sfn /lib/x86_64-linux-gnu/libusb-1.0.so.0 "$RUNTIME/libob_usb.so"
  SDK_LIB="$RUNTIME"
fi

mkdir -p "$(dirname "$OUTPUT")"

g++ -std=c++17 -O2 -pthread \
  -I"$SDK_ROOT/include" \
  "$ROOT/tools/orbbec_bag_unpack.cpp" \
  -L"$SDK_LIB" \
  -Wl,-rpath,"$SDK_LIB" \
  -lOrbbecSDK \
  -o "$OUTPUT"

echo "$OUTPUT"
