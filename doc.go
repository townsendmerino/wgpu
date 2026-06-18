// Package wgpu is a minimal, compute-only CGO binding over wgpu-native
// v29.0.0.0.
//
// It exists to give goinfer's ./gpu package a drop-in replacement for the
// slice of github.com/cogentcore/webgpu it uses, with full control over the
// WGSL dot4I8Packed builtin and the wgpu-native version. The exported type,
// method, and descriptor names mirror cogentcore/webgpu so migrating goinfer is
// a near-mechanical import swap (s,github.com/cogentcore/webgpu/wgpu,github.com/townsendmerino/wgpu,).
//
// Why CGO and not the zero-CGO goffi path (go-webgpu): goffi targets a fixed
// Go runtime crosscall2 callback ABI (Go 1.25) and SIGABRTs at RequestAdapter
// on Go 1.26. cgo callbacks are robust across Go versions.
//
// v29's webgpu.h is fully async (WGPUFuture + WGPUCallbackInfo for
// RequestAdapter / RequestDevice / MapAsync). This package hides that behind
// synchronous wrappers: adapter/device requests are driven to completion in C
// (wgpuInstanceProcessEvents) and buffer maps complete during Device.Poll, so
// the surface matches cogentcore's blocking style.
//
// Beyond the cogentcore drop-in subset this package also exposes v29-only
// capabilities useful for the dot4I8Packed work: GPU timestamp queries (for
// honest on-device DP4A measurement), pipeline-overridable WGSL constants,
// push-constant-equivalent "immediates", and subgroup adapter info.
//
// All struct layouts are transcribed from the webgpu.h / wgpu.h that ship
// inside the v29.0.0.0 release archives (vendored under lib/), which match the
// linked libwgpu_native.a. A wrong field offset is silent memory corruption,
// not a compile error — see the probe in cmd/dot4probe for cross-checking.
package wgpu
