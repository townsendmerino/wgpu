//go:build cgo

package wgpu_test

import (
	"testing"

	"github.com/townsendmerino/wgpu"
)

// These benchmarks isolate the CPU/CGO-side cost of the wrapper — the per-
// dispatch record path, CreateBindGroup, and submit+poll — independent of GPU
// kernel execution time. They quantify where (if anywhere) the wrapper itself
// is worth optimizing. Run: go test -run x -bench Wrapper -benchmem ./...

const benchWGSL = `
@group(0) @binding(0) var<storage, read>       inb:  array<u32>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&outb)) { return; }
    outb[i] = inb[i] + 1u;
}
`

func benchSetup(b *testing.B) (*wgpu.Instance, *wgpu.Adapter, *wgpu.Device, *wgpu.ComputePipeline, *wgpu.BindGroup, *wgpu.Queue) {
	b.Helper()
	inst := wgpu.CreateInstance(nil)
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	if err != nil {
		inst.Release()
		b.Skipf("no GPU adapter: %v", err)
	}
	dev, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredLimits: &wgpu.RequiredLimits{Limits: wgpu.DefaultLimits()}})
	if err != nil {
		adapter.Release()
		inst.Release()
		b.Fatalf("request device: %v", err)
	}
	sh, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: benchWGSL}})
	if err != nil {
		b.Fatalf("shader: %v", err)
	}
	pl, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"}})
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}
	sh.Release()
	a, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: 1024, Usage: wgpu.BufferUsageStorage})
	c, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: 1024, Usage: wgpu.BufferUsageStorage})
	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pl.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: a, Size: a.GetSize()},
			{Binding: 1, Buffer: c, Size: c.GetSize()},
		},
	})
	if err != nil {
		b.Fatalf("bindgroup: %v", err)
	}
	return inst, adapter, dev, pl, bg, dev.GetQueue()
}

// BenchmarkWrapperRecordPass measures the CPU/CGO cost to RECORD + submit (no
// poll) a compute pass of K dispatches — the "chain a step" cost, isolated from
// the GPU's own execution by not blocking on completion. ns/op ÷ K ≈
// per-dispatch wrapper cost. Submitting (vs. record-only) also avoids Metal's
// 4096 outstanding-command-buffer cap.
func BenchmarkWrapperRecordPass(b *testing.B) {
	inst, adapter, dev, pl, bg, q := benchSetup(b)
	defer func() { q.Release(); bg.Release(); pl.Release(); dev.Release(); adapter.Release(); inst.Release() }()
	const K = 512
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		for range K {
			pass.SetBindGroup(0, bg, nil)
			pass.DispatchWorkgroups(1, 1, 1)
		}
		pass.End()
		pass.Release()
		cmd, _ := enc.Finish(nil)
		q.Submit(cmd)
		cmd.Release()
		enc.Release()
	}
	dev.Poll(true, nil) // drain queued work once at the end
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/K, "ns/dispatch")
}

// BenchmarkWrapperCreateBindGroup measures CreateBindGroup+Release (calloc/free
// + 2 mutex ops + 1 cgo call). Relevant if a caller rebuilds bind groups/dispatch.
func BenchmarkWrapperCreateBindGroup(b *testing.B) {
	inst, adapter, dev, pl, bg0, q := benchSetup(b)
	defer func() { q.Release(); bg0.Release(); pl.Release(); dev.Release(); adapter.Release(); inst.Release() }()
	a, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: 1024, Usage: wgpu.BufferUsageStorage})
	defer a.Release()
	c, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: 1024, Usage: wgpu.BufferUsageStorage})
	defer c.Release()
	layout := pl.GetBindGroupLayout(0)
	entries := []wgpu.BindGroupEntry{{Binding: 0, Buffer: a, Size: a.GetSize()}, {Binding: 1, Buffer: c, Size: c.GetSize()}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: entries})
		if err != nil {
			b.Fatal(err)
		}
		bg.Release()
	}
}

// BenchmarkWrapperSubmitPoll measures one submit+blocking-poll round trip
// (1-dispatch command buffer). Captures the synchronous Submit+Poll overhead.
func BenchmarkWrapperSubmitPoll(b *testing.B) {
	inst, adapter, dev, pl, bg, q := benchSetup(b)
	defer func() { q.Release(); bg.Release(); pl.Release(); dev.Release(); adapter.Release(); inst.Release() }()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		enc, _ := dev.CreateCommandEncoder(nil)
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		pass.SetBindGroup(0, bg, nil)
		pass.DispatchWorkgroups(1, 1, 1)
		pass.End()
		pass.Release()
		cmd, _ := enc.Finish(nil)
		q.Submit(cmd)
		cmd.Release()
		enc.Release()
		dev.Poll(true, nil)
	}
}
