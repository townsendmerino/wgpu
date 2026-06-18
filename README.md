# wgpu — minimal compute-only Go binding over wgpu-native v29.0.0.0

A small **CGO** binding over [wgpu-native](https://github.com/gfx-rs/wgpu-native)
**v29.0.0.0**, built for one purpose: full control over the WGSL
`dot4I8Packed` builtin from Go, with an API that is a **drop-in for the slice of
[`github.com/cogentcore/webgpu`](https://github.com/cogentcore/webgpu) that
goinfer's `./gpu` package uses**. Migrating goinfer is a near-mechanical import
swap.

## Why this exists

`go-webgpu` (the zero-CGO `goffi` binding) SIGABRTs at `RequestAdapter` on Go
1.26: `goffi` targets the Go 1.25 `crosscall2` callback ABI, which broke. **cgo
callbacks are robust across Go versions.** This binding uses cgo and statically
links a prebuilt `libwgpu_native.a` into a single binary.

## Status

Built and validated on **darwin/arm64 (Apple M1 Pro, Metal)** against the real
v29.0.0.0 static lib:

```
adapter: Apple M1 Pro | backend=Metal | type=IntegratedGPU
ABI validation: PASS (dot4I8Packed results match CPU reference)
dot4I8Packed :  38.7 Gdot4/s
scalar       :  12.3 Gdot4/s
speedup (scalar/dot4): 3.16x   →  GO ✅
```

`dot4I8Packed` needs **no device feature** for correctness (naga polyfills it
everywhere). There is **no packed-dot feature flag** in wgpu-native v29 — the
DP4A fast path is selected automatically by wgpu-core/naga. Detection is
empirical: compile the builtin and **measure** (see `cmd/dot4probe`).

## Layout

```
*.go                       the binding (package wgpu)
lib/<goos>/<goarch>/libwgpu_native.a   vendored static libs
lib/webgpu.h, lib/wgpu.h   headers from the v29.0.0.0 release (match the libs)
lib/licenses/              wgpu-native MIT + Apache-2.0 texts
scripts/fetch.sh           re-vendor the libs/headers (build works offline after)
cmd/dot4probe/             Phase-2 go/no-go: ABI validation + DP4A measurement
```

Vendored platforms: `darwin/arm64`, `darwin/amd64`, `linux/amd64`,
`linux/arm64`, `windows/amd64` (GNU). Re-fetch with `bash scripts/fetch.sh`.

## Build & run

```sh
CGO_ENABLED=1 go build ./...
CGO_ENABLED=1 go test ./...          # ABI correctness test (skips without a GPU)
CGO_ENABLED=1 go run ./cmd/dot4probe # full DP4A measurement
```

## Migrating goinfer

goinfer's `./gpu` imports `github.com/cogentcore/webgpu/wgpu` as `wgpu`. The
exported type, method, and descriptor names here match that subset, so:

```sh
# in goinfer/gpu
grep -rl 'cogentcore/webgpu/wgpu' . | xargs sed -i '' \
  's#github.com/cogentcore/webgpu/wgpu#github.com/townsendmerino/wgpu#g'
```

The import alias stays `wgpu`, every `wgpu.X` call site is unchanged, and the
blocking call style (`RequestAdapter` returns `(*Adapter, error)`; `MapAsync` +
`Poll(true, nil)`) is preserved. v29's async futures are hidden behind
synchronous wrappers.

## Beyond the drop-in (v29 extras)

The cogentcore surface is mirrored exactly; on top of it this binding also
exposes v29-only capabilities useful for the dot4 work:

| Feature | API |
|---|---|
| GPU timestamp queries | `Device.CreateQuerySet`, `ComputePassDescriptor.TimestampWrites`, `CommandEncoder.ResolveQuerySet`, `Queue.GetTimestampPeriod` |
| Pipeline-overridable WGSL constants | `ProgrammableStageDescriptor.Constants []ConstantEntry` |
| Push-constant-equivalent immediates | `ComputePassEncoder.SetImmediates`, `NativeFeatureImmediates`, `Limits.MaxPushConstantSize` (→ `maxImmediateSize`) |
| Subgroup adapter info | `AdapterInfo.SubgroupMinSize/MaxSize`, `FeatureNameSubgroups`, `NativeFeatureSubgroup` |
| Batched dispatch recording (one CGO crossing for a whole chain; ~5× faster record) | `ComputePassEncoder.RecordSteps([]ComputeStep)` |

## License

MIT (this binding). Vendored wgpu-native blobs are MIT OR Apache-2.0 — see
`NOTICE` and `lib/licenses/`.
