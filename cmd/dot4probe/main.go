// Command dot4probe is the Phase-2 go/no-go: it brings up a wgpu-native v29
// device through this binding, validates the struct ABI with a correctness
// check, and measures dot4I8Packed throughput against a scalar polyfill to
// detect whether the hardware DP4A path engages.
//
// There is NO feature flag for packed integer dot product in wgpu-native v29 —
// the DP4A fast path is selected automatically by wgpu-core/naga. Detection is
// therefore empirical: compile the builtin and MEASURE.
//
// Usage:
//
//	go run -tags cgo ./cmd/dot4probe
package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/townsendmerino/wgpu"
)

const workgroupSize = 256

// Correctness kernel: out[i] = dot4I8Packed(a[i], b[i]).
const correctnessWGSL = `
@group(0) @binding(0) var<storage, read>       a:   array<u32>;
@group(0) @binding(1) var<storage, read>       b:   array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = dot4I8Packed(a[i], b[i]);
}
`

// Throughput kernels. Each thread runs a dependent chain of `iters` packed dot
// products; mutating bv by acc each step defeats loop-hoisting so the ALU work
// is real and sequential. Two variants share the exact same structure so the
// only difference is dot4I8Packed (HW DP4A candidate) vs a scalar polyfill.
const benchHeader = `
struct P { n: u32, iters: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read>       a:   array<u32>;
@group(0) @binding(1) var<storage, read>       b:   array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@group(0) @binding(3) var<uniform>             p:   P;
`

const benchDot4Body = `
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let av = a[i];
    var bv = b[i];
    var acc: i32 = 0;
    for (var k: u32 = 0u; k < p.iters; k = k + 1u) {
        acc = acc + dot4I8Packed(av, bv);
        bv = bv + u32(acc);
    }
    out[i] = acc;
}
`

const benchScalarBody = `
fn dotManual(x: u32, y: u32) -> i32 {
    var s: i32 = 0;
    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
        let sh = 8u * j;
        let xb = i32((x >> sh) & 0xffu);
        let yb = i32((y >> sh) & 0xffu);
        let xs = (xb ^ 0x80) - 0x80;   // sign-extend 8-bit
        let ys = (yb ^ 0x80) - 0x80;
        s = s + xs * ys;
    }
    return s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.n) { return; }
    let av = a[i];
    var bv = b[i];
    var acc: i32 = 0;
    for (var k: u32 = 0u; k < p.iters; k = k + 1u) {
        acc = acc + dotManual(av, bv);
        bv = bv + u32(acc);
    }
    out[i] = acc;
}
`

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "dot4probe: FAIL:", err)
		os.Exit(1)
	}
}

func run() error {
	fmt.Printf("wgpu-native version: 0x%08x\n", wgpu.GetVersion())

	inst := wgpu.CreateInstance(nil)
	defer inst.Release()

	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		return fmt.Errorf("request adapter: %w", err)
	}
	defer adapter.Release()

	info := adapter.GetInfo()
	fmt.Printf("adapter: %s | backend=%s | type=%s | vendor=%s\n",
		info.Name, info.BackendType, info.AdapterType, info.VendorName)
	fmt.Printf("subgroup size: min=%d max=%d\n", info.SubgroupMinSize, info.SubgroupMaxSize)

	hasTimestamp := adapter.HasFeature(wgpu.FeatureNameTimestampQuery)
	fmt.Printf("timestamp-query feature: %v\n", hasTimestamp)

	// Request device, raising buffer/binding limits like goinfer does.
	lim := wgpu.DefaultLimits()
	al := adapter.GetLimits().Limits
	lim.MaxStorageBufferBindingSize = al.MaxStorageBufferBindingSize
	lim.MaxBufferSize = al.MaxStorageBufferBindingSize
	if al.MaxComputeWorkgroupsPerDimension > lim.MaxComputeWorkgroupsPerDimension {
		lim.MaxComputeWorkgroupsPerDimension = al.MaxComputeWorkgroupsPerDimension
	}
	dd := &wgpu.DeviceDescriptor{RequiredLimits: &wgpu.RequiredLimits{Limits: lim}}
	if hasTimestamp {
		dd.RequiredFeatures = []wgpu.FeatureName{wgpu.FeatureNameTimestampQuery}
	}
	device, err := adapter.RequestDevice(dd)
	if err != nil {
		return fmt.Errorf("request device: %w", err)
	}
	defer device.Release()
	queue := device.GetQueue()
	defer queue.Release()

	// ---- ABI validation: correctness cross-check ----------------------------
	if err := validate(device, queue); err != nil {
		return fmt.Errorf("ABI validation: %w", err)
	}
	fmt.Println("\nABI validation: PASS (dot4I8Packed results match CPU reference)")

	// ---- DP4A measurement ---------------------------------------------------
	if err := measure(device, queue, hasTimestamp); err != nil {
		return fmt.Errorf("measure: %w", err)
	}
	return nil
}

// cpuDot4 is the reference: signed 8-bit lanes packed little-endian in a u32.
func cpuDot4(x, y uint32) int32 {
	var s int32
	for j := range 4 {
		xs := int32(int8(x >> (8 * j)))
		ys := int32(int8(y >> (8 * j)))
		s += xs * ys
	}
	return s
}

func validate(device *wgpu.Device, queue *wgpu.Queue) error {
	const n = 4096
	a := make([]uint32, n)
	b := make([]uint32, n)
	rng := rand.New(rand.NewSource(1))
	for i := range a {
		a[i] = rng.Uint32()
		b[i] = rng.Uint32()
	}

	sh, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "dot4-correctness",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: correctnessWGSL},
	})
	if err != nil {
		return err
	}
	defer sh.Release()
	pl, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "dot4-correctness",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}
	defer pl.Release()

	aBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "a", Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return err
	}
	defer aBuf.Release()
	bBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "b", Contents: wgpu.ToBytes(b), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return err
	}
	defer bBuf.Release()
	outBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "out", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return err
	}
	defer outBuf.Release()
	stage, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "stage", Size: uint64(n * 4), Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return err
	}
	defer stage.Release()

	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pl.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 2, Buffer: outBuf, Size: outBuf.GetSize()},
		},
	})
	if err != nil {
		return err
	}
	defer bg.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((n+workgroupSize-1)/workgroupSize, 1, 1)
	if err := pass.End(); err != nil {
		return err
	}
	pass.Release()
	if err := enc.CopyBufferToBuffer(outBuf, 0, stage, 0, uint64(n*4)); err != nil {
		return err
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	queue.Submit(cmd)
	cmd.Release()

	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, uint64(n*4), func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return err
	}
	device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return fmt.Errorf("map failed: %v", status)
	}
	got := wgpu.FromBytes[int32](stage.GetMappedRange(0, uint(n*4)))
	mism := 0
	for i := range n {
		if got[i] != cpuDot4(a[i], b[i]) {
			if mism < 5 {
				fmt.Printf("  mismatch at %d: gpu=%d cpu=%d (a=%08x b=%08x)\n", i, got[i], cpuDot4(a[i], b[i]), a[i], b[i])
			}
			mism++
		}
	}
	if err := stage.Unmap(); err != nil {
		return err
	}
	if mism > 0 {
		return fmt.Errorf("%d/%d mismatches (likely struct ABI/offset bug)", mism, n)
	}
	return nil
}

func measure(device *wgpu.Device, queue *wgpu.Queue, hasTimestamp bool) error {
	const (
		n     = 1 << 20 // 1M threads
		iters = 4096    // dependent dot products per thread
		runs  = 50
	)
	fmt.Printf("\nDP4A measurement: n=%d threads, iters=%d/thread, runs=%d\n", n, iters, runs)
	totalOps := float64(n) * float64(iters) * float64(runs)

	dot4ns, err := benchKernel(device, queue, benchHeader+benchDot4Body, "dot4I8Packed", n, iters, runs, hasTimestamp)
	if err != nil {
		return err
	}
	scalarns, err := benchKernel(device, queue, benchHeader+benchScalarBody, "scalar-polyfill", n, iters, runs, hasTimestamp)
	if err != nil {
		return err
	}

	dot4Gops := totalOps / dot4ns
	scalarGops := totalOps / scalarns
	fmt.Printf("\n  dot4I8Packed : %8.2f ms total  | %7.1f Gdot4/s\n", dot4ns/1e6, dot4Gops)
	fmt.Printf("  scalar       : %8.2f ms total  | %7.1f Gdot4/s\n", scalarns/1e6, scalarGops)
	speedup := scalarns / dot4ns
	fmt.Printf("  speedup (scalar/dot4): %.2fx\n", speedup)

	fmt.Println("\nVERDICT:")
	switch {
	case speedup >= 1.8:
		fmt.Printf("  GO ✅  dot4I8Packed is %.2fx faster than the scalar polyfill — the hardware\n", speedup)
		fmt.Println("        DP4A path is engaging. This binding gives you the fast packed-dot kernel.")
	case speedup >= 1.15:
		fmt.Printf("  LEAN-GO 🟡  dot4I8Packed is %.2fx faster — a real but modest win. Likely a\n", speedup)
		fmt.Println("        vectorized/optimized path rather than a dedicated DP4A instruction on this GPU.")
	default:
		fmt.Printf("  NO-GO ⚠️  dot4I8Packed is only %.2fx vs scalar — naga is emitting a polyfill on\n", speedup)
		fmt.Println("        this adapter/backend; no DP4A acceleration here. (Correctness still holds.)")
	}
	return nil
}

// benchKernel runs the given compute kernel `runs` times and returns total GPU
// time in nanoseconds. Uses GPU timestamps when available, else wall-clock.
func benchKernel(device *wgpu.Device, queue *wgpu.Queue, code, label string, n, iters, runs int, hasTimestamp bool) (float64, error) {
	sh, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: label, WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: code},
	})
	if err != nil {
		return 0, err
	}
	defer sh.Release()
	pl, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: label, Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		return 0, err
	}
	defer pl.Release()

	a := make([]uint32, n)
	b := make([]uint32, n)
	rng := rand.New(rand.NewSource(2))
	for i := range a {
		a[i] = rng.Uint32()
		b[i] = rng.Uint32()
	}
	aBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "a", Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return 0, err
	}
	defer aBuf.Release()
	bBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "b", Contents: wgpu.ToBytes(b), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return 0, err
	}
	defer bBuf.Release()
	outBuf, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: "out", Size: uint64(n * 4), Usage: wgpu.BufferUsageStorage})
	if err != nil {
		return 0, err
	}
	defer outBuf.Release()
	params := []uint32{uint32(n), uint32(iters), 0, 0}
	pBuf, err := device.CreateBufferInit(&wgpu.BufferInitDescriptor{Label: "p", Contents: wgpu.ToBytes(params), Usage: wgpu.BufferUsageUniform})
	if err != nil {
		return 0, err
	}
	defer pBuf.Release()

	bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pl.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 2, Buffer: outBuf, Size: outBuf.GetSize()},
			{Binding: 3, Buffer: pBuf, Size: pBuf.GetSize()},
		},
	})
	if err != nil {
		return 0, err
	}
	defer bg.Release()

	// Optional timestamp plumbing.
	var qset *wgpu.QuerySet
	var tsResolve, tsStage *wgpu.Buffer
	if hasTimestamp {
		qset, err = device.CreateQuerySet(&wgpu.QuerySetDescriptor{Label: "ts", Type: wgpu.QueryTypeTimestamp, Count: 2})
		if err != nil {
			return 0, err
		}
		defer qset.Release()
		tsResolve, err = device.CreateBuffer(&wgpu.BufferDescriptor{Label: "tsResolve", Size: 16, Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc})
		if err != nil {
			return 0, err
		}
		defer tsResolve.Release()
		tsStage, err = device.CreateBuffer(&wgpu.BufferDescriptor{Label: "tsStage", Size: 16, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return 0, err
		}
		defer tsStage.Release()
	}

	groups := uint32((n + workgroupSize - 1) / workgroupSize)
	period := float64(1.0)
	if hasTimestamp {
		period = float64(queue.GetTimestampPeriod())
	}

	// Warmup.
	if err := dispatchOnce(device, queue, pl, bg, groups, nil); err != nil {
		return 0, err
	}

	var gpuNS float64
	wallStart := time.Now()
	for range runs {
		var tw *wgpu.ComputePassTimestampWrites
		if hasTimestamp {
			tw = &wgpu.ComputePassTimestampWrites{QuerySet: qset, BeginningOfPassWriteIndex: 0, EndOfPassWriteIndex: 1}
		}
		if err := dispatchOnce(device, queue, pl, bg, groups, tw); err != nil {
			return 0, err
		}
		if hasTimestamp {
			ts, err := readTimestamps(device, queue, qset, tsResolve, tsStage)
			if err != nil {
				return 0, err
			}
			gpuNS += float64(ts[1]-ts[0]) * period
		}
	}
	wallNS := float64(time.Since(wallStart).Nanoseconds())

	if hasTimestamp && gpuNS > 0 {
		fmt.Printf("  %-16s GPU(timestamps)=%.2f ms  wall=%.2f ms\n", label, gpuNS/1e6, wallNS/1e6)
		return gpuNS, nil
	}
	fmt.Printf("  %-16s wall=%.2f ms (no timestamps)\n", label, wallNS/1e6)
	return wallNS, nil
}

func dispatchOnce(device *wgpu.Device, queue *wgpu.Queue, pl *wgpu.ComputePipeline, bg *wgpu.BindGroup, groups uint32, tw *wgpu.ComputePassTimestampWrites) error {
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return err
	}
	defer enc.Release()
	var passDesc *wgpu.ComputePassDescriptor
	if tw != nil {
		passDesc = &wgpu.ComputePassDescriptor{TimestampWrites: tw}
	}
	pass := enc.BeginComputePass(passDesc)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups(groups, 1, 1)
	if err := pass.End(); err != nil {
		return err
	}
	pass.Release()
	cmd, err := enc.Finish(nil)
	if err != nil {
		return err
	}
	queue.Submit(cmd)
	cmd.Release()
	device.Poll(true, nil)
	return nil
}

func readTimestamps(device *wgpu.Device, queue *wgpu.Queue, qset *wgpu.QuerySet, resolve, stage *wgpu.Buffer) ([2]uint64, error) {
	var out [2]uint64
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return out, err
	}
	defer enc.Release()
	enc.ResolveQuerySet(qset, 0, 2, resolve, 0)
	if err := enc.CopyBufferToBuffer(resolve, 0, stage, 0, 16); err != nil {
		return out, err
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return out, err
	}
	queue.Submit(cmd)
	cmd.Release()
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, 16, func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return out, err
	}
	device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return out, fmt.Errorf("timestamp map failed: %v", status)
	}
	ts := wgpu.FromBytes[uint64](stage.GetMappedRange(0, 16))
	out[0], out[1] = ts[0], ts[1]
	if err := stage.Unmap(); err != nil {
		return out, err
	}
	return out, nil
}
