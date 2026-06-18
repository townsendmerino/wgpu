//go:build cgo

package wgpu_test

import (
	"testing"

	"github.com/townsendmerino/wgpu"
)

// newDevice brings up a device or skips the test if no adapter is available
// (e.g. a headless CI box) — matching how goinfer's gpu tests degrade.
func newDevice(t *testing.T) (*wgpu.Instance, *wgpu.Adapter, *wgpu.Device) {
	t.Helper()
	inst := wgpu.CreateInstance(nil)
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		inst.Release()
		t.Skipf("no GPU adapter available: %v", err)
	}
	dev, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{Limits: wgpu.DefaultLimits()},
	})
	if err != nil {
		adapter.Release()
		inst.Release()
		t.Fatalf("request device: %v", err)
	}
	return inst, adapter, dev
}

const dot4WGSL = `
@group(0) @binding(0) var<storage, read>       a:   array<u32>;
@group(0) @binding(1) var<storage, read>       b:   array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) { return; }
    out[i] = dot4I8Packed(a[i], b[i]);
}
`

func cpuDot4(x, y uint32) int32 {
	var s int32
	for j := 0; j < 4; j++ {
		s += int32(int8(x>>(8*j))) * int32(int8(y>>(8*j)))
	}
	return s
}

// TestDot4Correctness is the ABI cross-check: GPU dot4I8Packed must match the
// CPU reference. A struct-offset bug would corrupt the results.
func TestDot4Correctness(t *testing.T) {
	inst, adapter, dev := newDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer dev.Release()
	queue := dev.GetQueue()
	defer queue.Release()

	const n = 1024
	a := make([]uint32, n)
	b := make([]uint32, n)
	for i := range a {
		a[i] = uint32(i*2654435761) ^ 0x9e3779b9
		b[i] = uint32(i*40503) ^ 0x85ebca6b
	}

	sh, err := dev.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "dot4", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: dot4WGSL},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer sh.Release()
	pl, err := dev.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "dot4", Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer pl.Release()

	aBuf, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(a), Usage: wgpu.BufferUsageStorage})
	defer aBuf.Release()
	bBuf, _ := dev.CreateBufferInit(&wgpu.BufferInitDescriptor{Contents: wgpu.ToBytes(b), Usage: wgpu.BufferUsageStorage})
	defer bBuf.Release()
	outBuf, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: n * 4, Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc})
	defer outBuf.Release()
	stage, _ := dev.CreateBuffer(&wgpu.BufferDescriptor{Size: n * 4, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	defer stage.Release()

	bg, err := dev.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: pl.GetBindGroupLayout(0),
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 2, Buffer: outBuf, Size: outBuf.GetSize()},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer bg.Release()

	enc, _ := dev.CreateCommandEncoder(nil)
	defer enc.Release()
	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(pl)
	pass.SetBindGroup(0, bg, nil)
	pass.DispatchWorkgroups((n+63)/64, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatal(err)
	}
	pass.Release()
	if err := enc.CopyBufferToBuffer(outBuf, 0, stage, 0, n*4); err != nil {
		t.Fatal(err)
	}
	cmd, _ := enc.Finish(nil)
	queue.Submit(cmd)
	cmd.Release()

	status := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, n*4, func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		t.Fatal(err)
	}
	dev.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		t.Fatalf("map failed: %v", status)
	}
	got := wgpu.FromBytes[int32](stage.GetMappedRange(0, n*4))
	for i := 0; i < n; i++ {
		if want := cpuDot4(a[i], b[i]); got[i] != want {
			t.Fatalf("dot4 mismatch at %d: got %d want %d (ABI/offset bug?)", i, got[i], want)
		}
	}
	stage.Unmap()
}

func TestByteHelpers(t *testing.T) {
	in := []uint32{1, 2, 0xdeadbeef, 0}
	round := wgpu.FromBytes[uint32](wgpu.ToBytes(in))
	if len(round) != len(in) {
		t.Fatalf("len %d != %d", len(round), len(in))
	}
	for i := range in {
		if round[i] != in[i] {
			t.Fatalf("roundtrip %d: %x != %x", i, round[i], in[i])
		}
	}
}
