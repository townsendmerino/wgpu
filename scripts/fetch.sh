#!/usr/bin/env bash
# Fetch & vendor wgpu-native v29.0.0.0 static libs + matching headers.
# Idempotent: re-running re-downloads only missing archives.
# After running, the build works fully offline.
set -euo pipefail

VERSION="v29.0.0.0"
BASE="https://github.com/gfx-rs/wgpu-native/releases/download/${VERSION}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${ROOT}/lib"
TMP="${ROOT}/.fetch-tmp"
mkdir -p "${LIB}" "${TMP}"

# release-asset-stem  ->  goos/goarch
MAP=(
  "wgpu-macos-aarch64-release:darwin/arm64"
  "wgpu-macos-x86_64-release:darwin/amd64"
  "wgpu-linux-x86_64-release:linux/amd64"
  "wgpu-linux-aarch64-release:linux/arm64"
  "wgpu-windows-x86_64-gnu-release:windows/amd64"
)

HDR_DONE=0
for entry in "${MAP[@]}"; do
  stem="${entry%%:*}"
  dest="${entry##*:}"
  zip="${TMP}/${stem}.zip"
  out="${LIB}/${dest}"
  mkdir -p "${out}"

  if [[ -f "${out}/libwgpu_native.a" ]]; then
    echo "✓ ${dest}/libwgpu_native.a present, skipping"
  else
    echo "↓ ${stem}.zip"
    curl -fL --retry 3 -o "${zip}" "${BASE}/${stem}.zip"
    rm -rf "${TMP}/x"; mkdir -p "${TMP}/x"
    unzip -q -o "${zip}" -d "${TMP}/x"
    cp "${TMP}/x/lib/libwgpu_native.a" "${out}/libwgpu_native.a"
    echo "  -> ${out}/libwgpu_native.a"
  fi

  # Vendor the headers from the FIRST archive that has them — they must match the lib.
  if [[ "${HDR_DONE}" -eq 0 && -d "${TMP}/x/include" ]]; then
    cp "${TMP}/x/include/webgpu/webgpu.h" "${LIB}/webgpu.h"
    cp "${TMP}/x/include/webgpu/wgpu.h"   "${LIB}/wgpu.h"
    echo "  -> vendored webgpu.h, wgpu.h from ${stem}"
    HDR_DONE=1
  fi
done

# Fallback: if headers weren't captured (all libs were cached), grab macos arm64 just for headers.
if [[ ! -f "${LIB}/webgpu.h" ]]; then
  echo "↓ fetching headers (macos-aarch64)"
  curl -fL --retry 3 -o "${TMP}/hdr.zip" "${BASE}/wgpu-macos-aarch64-release.zip"
  rm -rf "${TMP}/h"; mkdir -p "${TMP}/h"
  unzip -q -o "${TMP}/hdr.zip" -d "${TMP}/h"
  cp "${TMP}/h/include/webgpu/webgpu.h" "${LIB}/webgpu.h"
  cp "${TMP}/h/include/webgpu/wgpu.h"   "${LIB}/wgpu.h"
fi

rm -rf "${TMP}"
echo "done. wgpu-native ${VERSION} vendored under ${LIB}"
