#!/usr/bin/env bash
# Version bisect for the dependent-dispatch-chain regression (STEP 2).
#
# Builds the SAME self-contained cmd/chainprobe against successive wgpu-native
# prebuilt releases (linux x86_64) by swapping lib/{webgpu.h,wgpu.h} and
# lib/linux/amd64/libwgpu_native.a per release, then runs it and prints the
# per-version CSV rows. The probe uses only the stable C-API subset, so the same
# source compiles against every release given two compile flags (see below).
#
# Usage (on the Linux/Vulkan box, from the repo root):
#   GO=$HOME/sdk/go/bin/go BACKEND=vulkan KS=100,200,400 bash scripts/bisect.sh
#
# Env:
#   GO        path to go (default: go on PATH)
#   BACKEND   vulkan|gl|metal (default: vulkan)
#   KS        comma-separated chain lengths (default: 100,200,400)
#   N, RUNS   elems/dispatch, timed reps (default: 256, 30)
#   VERSIONS  space-separated tags (default: v29.0.0.0 v27.0.4.1 v25.0.2.2)
#   TS_PERIOD ns/tick for releases lacking wgpuQueueGetTimestampPeriod (v25).
#             NVIDIA Vulkan reports 1.0; override if your GPU differs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GO="${GO:-go}"
BACKEND="${BACKEND:-vulkan}"
KS="${KS:-100,200,400}"
N="${N:-256}"
RUNS="${RUNS:-30}"
TS_PERIOD="${TS_PERIOD:-1.0}"
VERSIONS="${VERSIONS:-v29.0.0.0 v27.0.4.1 v25.0.2.2}"
CACHE="${CACHE:-$HOME/.cache/wgpu-bisect}"
LIB="${ROOT}/lib"
STEM="wgpu-linux-x86_64-release"
BASE="https://github.com/gfx-rs/wgpu-native/releases/download"

# Per-release compile flags for the two ABI deltas in the stable subset:
#   v25: timestamp-writes struct is WGPUComputePassTimestampWrites (renamed to
#        WGPUPassTimestampWrites in v27), and there is no
#        wgpuQueueGetTimestampPeriod (added in v27).
flags_for() {
  case "$1" in
    v25.*) echo "-DWGPU_TS_LEGACY -DWGPU_NO_TS_PERIOD -DTS_PERIOD=${TS_PERIOD}f" ;;
    *)     echo "" ;;
  esac
}

mkdir -p "${CACHE}"

# Back up whatever is currently vendored so we can restore it at the end.
BAK="$(mktemp -d)"
cp "${LIB}/webgpu.h" "${LIB}/wgpu.h" "${BAK}/" 2>/dev/null || true
cp "${LIB}/linux/amd64/libwgpu_native.a" "${BAK}/libwgpu_native.a" 2>/dev/null || true
restore() {
  echo "restoring vendored lib..."
  cp "${BAK}/webgpu.h" "${BAK}/wgpu.h" "${LIB}/" 2>/dev/null || true
  cp "${BAK}/libwgpu_native.a" "${LIB}/linux/amd64/libwgpu_native.a" 2>/dev/null || true
  rm -rf "${BAK}"
}
trap restore EXIT

fetch() { # tag -> populates $CACHE/$tag/{webgpu.h,wgpu.h,libwgpu_native.a}
  local tag="$1" dir="${CACHE}/$1"
  if [[ -f "${dir}/libwgpu_native.a" && -f "${dir}/webgpu.h" ]]; then return; fi
  mkdir -p "${dir}"
  echo "↓ fetching ${tag} (${STEM})"
  curl -fL --retry 3 -o "${dir}/a.zip" "${BASE}/${tag}/${STEM}.zip"
  rm -rf "${dir}/x"; mkdir -p "${dir}/x"
  unzip -q -o "${dir}/a.zip" -d "${dir}/x"
  cp "${dir}/x/lib/libwgpu_native.a"           "${dir}/libwgpu_native.a"
  cp "${dir}/x/include/webgpu/webgpu.h"        "${dir}/webgpu.h"
  cp "${dir}/x/include/webgpu/wgpu.h"          "${dir}/wgpu.h"
  rm -rf "${dir}/x" "${dir}/a.zip"
}

echo "=== bisect: backend=${BACKEND} K=${KS} n=${N} runs=${RUNS} ==="
echo "csv,verHex,backend,mode,K,n,gpu_ms,wall_ms,per_dispatch_us,per_barrier_us"
for tag in ${VERSIONS}; do
  fetch "${tag}"
  cp "${CACHE}/${tag}/webgpu.h" "${CACHE}/${tag}/wgpu.h" "${LIB}/"
  cp "${CACHE}/${tag}/libwgpu_native.a" "${LIB}/linux/amd64/libwgpu_native.a"
  fl="$(flags_for "${tag}")"
  echo ">>> building chainprobe against ${tag} (CGO_CFLAGS='${fl}')" >&2
  CGO_ENABLED=1 CGO_CFLAGS="${fl}" "${GO}" build -o "/tmp/chainprobe-${tag}" ./cmd/chainprobe
  "/tmp/chainprobe-${tag}" -backend "${BACKEND}" -ksweep "${KS}" -n "${N}" -runs "${RUNS}" -csv \
    | sed -n 's/^csv,//p' | sed "s/^/csv,/"
  echo >&2
done
echo "=== bisect done ===" >&2
