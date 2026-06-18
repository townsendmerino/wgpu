#!/usr/bin/env bash
# Reproduce the naga-codegen A/B SPIR-V blobs (bench/nagaprobe/spv/*.spv).
#
# naga-cli is published per wgpu version. wgpu-native v22.1.0.5 (cogentcore's
# bundled v22-era lib) pins naga 22.1.0; wgpu-native v29.0.0.0 pins naga 29.0.1.
# The closest published naga-cli releases are 22.0.0 and 29.0.0/29.0.2 — codegen
# differs only at the patch level within a major, and nagaprobe's WGSL "runtime"
# arm cross-checks against the runtime's actual naga 29.0.1.
#
#   rustup default stable
#   cargo install naga-cli --version 22.0.0 --locked --root ~/naga22
#   cargo install naga-cli --version 29.0.2 --locked --root ~/naga29
#
# Then compile each fixture (compute entry "main") to SPIR-V with both:
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N22="${N22:-$HOME/naga22/bin/naga}"
N29="${N29:-$HOME/naga29/bin/naga}"
FX="${ROOT}/bench/nagaprobe/fixtures"
OUT="${ROOT}/bench/nagaprobe/spv"
mkdir -p "${OUT}"
for f in rmsnorm qknorm residual swiglu; do
  "${N22}" "${FX}/${f}.wgsl" "${OUT}/${f}.22.spv"
  "${N29}" "${FX}/${f}.wgsl" "${OUT}/${f}.29.spv"
  echo "${f}: 22=$(stat -c%s "${OUT}/${f}.22.spv" 2>/dev/null || stat -f%z "${OUT}/${f}.22.spv")B  29=$(stat -c%s "${OUT}/${f}.29.spv" 2>/dev/null || stat -f%z "${OUT}/${f}.29.spv")B"
done

# A/B on a v29 / Vulkan box (device-side timestamps, K=400):
#   for f in rmsnorm qknorm residual swiglu; do
#     ./nagaprobe -backend vulkan -fixture $f -spv  bench/nagaprobe/spv/$f.22.spv -label naga22   -csv
#     ./nagaprobe -backend vulkan -fixture $f -spv  bench/nagaprobe/spv/$f.29.spv -label naga29   -csv
#     ./nagaprobe -backend vulkan -fixture $f -wgsl bench/nagaprobe/fixtures/$f.wgsl -label wgsl29rt -csv
#   done
