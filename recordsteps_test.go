//go:build cgo

package wgpu_test

import (
	"testing"

	"github.com/townsendmerino/wgpu"
)

// in-place increment: each dispatch does b[i] += 1, so K serially-dependent
// dispatches on one buffer leave b[i] == K (WebGPU orders dependent dispatches
// within a pass). Lets us assert RecordSteps == the unrolled per-call loop.
const incWGSL = `
@group(0) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&b)) { return; }
    b[i] = b[i] + 1u;
}
`

func incSetup(tb testing.TB) (*wgpu.Instance, *wgpu.Adapter, *wgpu.Device, *wgpu.Queue, *wgpu.ComputePipeline, *wgpu.BindGroup, *wgpu.Buffer) {
	tb.Helper()
	inst := wgpu.CreateInstance(nil)
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	if err != nil {
		inst.Release()
		tb.Skipf("no GPU adapter: %v", err)
	}
	dev, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredLimits: &wgpu.RequiredLimits{Limits: wgpu.DefaultLimits()}})
	if err != nil {
		adapter.Release()
		inst.Release()
		tb.Fatalf("device: %v", err)
	}
	sh, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: incWGSL}})
	if err != nil {
		tb.Fatalf("shader: %v", err)
	}
	pl, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"}})
	if err != nil {
		tb.Fatalf("pipeline: %v", err)
	}
	sh.Release()
	const n = 256
	buf, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: n * 4, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout:  pl.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: buf, Size: buf.GetSize()}},
	})
	if err != nil {
		tb.Fatalf("bindgroup: %v", err)
	}
	return inst, adapter, dev, dev.GetQueue(), pl, bg, buf
}

// TestRecordStepsMatchesLoop verifies RecordSteps is bit-identical to the
// unrolled SetPipeline/SetBindGroup/DispatchWorkgroups loop.
func TestRecordStepsMatchesLoop(t *testing.T) {
	inst, adapter, dev, q, pl, bg, buf := incSetup(t)
	defer func() { q.Release(); bg.Release(); buf.Release(); pl.Release(); dev.Release(); adapter.Release(); inst.Release() }()
	const n, K = 64, 50 // n == workgroup_size so one (1,1,1) dispatch covers all elements

	run := func(batched bool) []uint32 {
		// fresh, zero-initialised buffer + bind group per run (no cross-run state)
		rbuf, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: n * 4, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
		defer rbuf.Release()
		rbg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout:  pl.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{{Binding: 0, Buffer: rbuf, Size: rbuf.GetSize()}},
		})
		if err != nil {
			t.Fatalf("bindgroup: %v", err)
		}
		defer rbg.Release()
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		if batched {
			steps := make([]wgpu.ComputeStep, K)
			for i := range steps {
				steps[i] = wgpu.ComputeStep{Pipeline: pl, BindGroup: rbg, X: 1, Y: 1, Z: 1}
			}
			pass.RecordSteps(steps)
		} else {
			for range K {
				pass.SetPipeline(pl)
				pass.SetBindGroup(0, rbg, nil)
				pass.DispatchWorkgroups(1, 1, 1)
			}
		}
		pass.End()
		pass.Release()
		stag, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: n * 4, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		defer stag.Release()
		enc.CopyBufferToBuffer(rbuf, 0, stag, 0, n*4)
		cmd, _ := enc.Finish(nil)
		q.Submit(cmd)
		cmd.Release()
		enc.Release()
		st := wgpu.BufferMapAsyncStatusUnknown
		stag.MapAsync(wgpu.MapModeRead, 0, n*4, func(s wgpu.BufferMapAsyncStatus) { st = s })
		dev.Poll(true, nil)
		if st != wgpu.BufferMapAsyncStatusSuccess {
			t.Fatalf("map failed: %v", st)
		}
		out := make([]uint32, n)
		copy(out, wgpu.FromBytes[uint32](stag.GetMappedRange(0, n*4)))
		stag.Unmap()
		return out
	}

	manual := run(false)
	batched := run(true)
	for i := range manual {
		if manual[i] != uint32(K) {
			t.Fatalf("manual[%d] = %d, want %d", i, manual[i], K)
		}
		if batched[i] != manual[i] {
			t.Fatalf("RecordSteps[%d] = %d != manual %d", i, batched[i], manual[i])
		}
	}
}

// BenchmarkWrapperRecordLoopManual / RecordStepsBatched measure record-only cost
// (no Finish/Submit, so no GPU work and no Metal command-buffer accumulation) for
// K dispatches: the manual per-call loop vs the batched single-crossing path.
func benchRecordOnly(b *testing.B, batched bool) {
	inst, adapter, dev, q, pl, bg, buf := incSetup(b)
	defer func() { q.Release(); bg.Release(); buf.Release(); pl.Release(); dev.Release(); adapter.Release(); inst.Release() }()
	const K = 512
	steps := make([]wgpu.ComputeStep, K)
	for i := range steps {
		steps[i] = wgpu.ComputeStep{Pipeline: pl, BindGroup: bg, X: 1, Y: 1, Z: 1}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		if batched {
			pass.RecordSteps(steps)
		} else {
			for range K {
				pass.SetPipeline(pl)
				pass.SetBindGroup(0, bg, nil)
				pass.DispatchWorkgroups(1, 1, 1)
			}
		}
		pass.End()
		pass.Release()
		enc.Release() // no Finish/Submit: isolates record cost
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/K, "ns/dispatch")
}

func BenchmarkWrapperRecordLoopManual(b *testing.B)   { benchRecordOnly(b, false) }
func BenchmarkWrapperRecordStepsBatched(b *testing.B) { benchRecordOnly(b, true) }
