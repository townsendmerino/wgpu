// Command cogentbase is the v22-era "known-fast" baseline for the
// dependent-dispatch-chain investigation (STEP 2b). It runs the SAME chain as
// cmd/chainbench / cmd/chainprobe — K serially-dependent tiny compute
// dispatches (ping-pong, read-after-write each step => one barrier per
// dispatch) vs an independent control (disjoint outputs, no hazard) — but
// through github.com/cogentcore/webgpu, which bundles the pre-futures
// (v22-era) libwgpu_native that cogentcore pinned and goinfer compared v29
// against.
//
// cogentcore's ComputePassDescriptor.TimestampWrites is not wired, so device
// time is bracketed with encoder-level WriteTimestamp around the compute pass.
// Because per-barrier cost is reported as a DELTA (dependent-independent)/K,
// any constant timestamp/encoder offset cancels, making the per-barrier number
// directly comparable to chainprobe's across versions. No GetTimestampPeriod in
// this cogentcore release, so we use 1.0 ns/tick (NVIDIA Vulkan), matching the
// v25 probe build.
//
// Output is the same CSV shape as chainprobe (version reported as 0xCOGE0023):
//
//	csv,0xcoge0023,<backend>,<mode>,<K>,<n>,<gpu_ms>,<wall_ms>,<per_dispatch_us>,<per_barrier_us>
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cogentcore/webgpu/wgpu"
)

const workgroupSize = 64

const kernel = `
@group(0) @binding(0) var<storage, read>       inb:  array<u32>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&outb)) { return; }
    outb[i] = inb[i] + 1u;
}
`

const tsPeriod = 1.0 // NVIDIA Vulkan ns/tick (cogentcore has no GetTimestampPeriod)

func main() {
	k := flag.Int("k", 400, "chain length")
	n := flag.Int("n", 256, "elements per buffer")
	runs := flag.Int("runs", 30, "timed reps (median)")
	ksweep := flag.String("ksweep", "", "comma-separated K values (overrides -k)")
	mode := flag.String("mode", "both", "both|dependent|independent")
	csv := flag.Bool("csv", false, "emit csv rows")
	flag.Parse()

	if err := run(*k, *n, *runs, *ksweep, *mode, *csv); err != nil {
		fmt.Fprintln(os.Stderr, "cogentbase: FAIL:", err)
		os.Exit(1)
	}
}

func run(k, n, runs int, ksweep, mode string, csv bool) error {
	inst := wgpu.CreateInstance(nil)
	defer inst.Release()
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{PowerPreference: wgpu.PowerPreferenceHighPerformance})
	if err != nil {
		return fmt.Errorf("request adapter: %w", err)
	}
	defer adapter.Release()
	info := adapter.GetInfo()
	hasTS := adapter.HasFeature(wgpu.FeatureNameTimestampQuery)
	fmt.Printf("cogentcore/webgpu v0.23.0 (v22-era lib)\n")
	fmt.Printf("adapter: %s | backend=%v | type=%v | timestamps=%v\n", info.Name, info.BackendType, info.AdapterType, hasTS)

	dd := &wgpu.DeviceDescriptor{}
	if hasTS {
		dd.RequiredFeatures = []wgpu.FeatureName{wgpu.FeatureNameTimestampQuery}
	}
	device, err := adapter.RequestDevice(dd)
	if err != nil {
		return fmt.Errorf("request device: %w", err)
	}
	defer device.Release()
	queue := device.GetQueue()
	defer queue.Release()

	sh, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "chain", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: kernel},
	})
	if err != nil {
		return err
	}
	defer sh.Release()
	pl, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "chain",
		Compute: wgpu.ProgrammableStageDescriptor{Module: sh, EntryPoint: "main"},
	})
	if err != nil {
		return err
	}
	defer pl.Release()

	backendName := fmt.Sprintf("%v", info.BackendType)
	ks := []int{k}
	if strings.TrimSpace(ksweep) != "" {
		ks = ks[:0]
		for _, p := range strings.Split(ksweep, ",") {
			if p = strings.TrimSpace(p); p == "" {
				continue
			}
			v, err := strconv.Atoi(p)
			if err != nil {
				return fmt.Errorf("bad -ksweep %q: %w", p, err)
			}
			ks = append(ks, v)
		}
	}

	doDep := mode == "both" || mode == "dependent"
	doIndep := mode == "both" || mode == "independent"

	fmt.Printf("\nn=%d elems/dispatch, runs=%d\n", n, runs)
	fmt.Printf("%-12s %6s  %12s %12s  %14s %14s\n", "mode", "K", "gpu(ms)", "wall(ms)", "per-disp(us)", "per-barr(us)")

	for _, kk := range ks {
		var depG, indepG float64
		if doDep {
			g, w, err := bench(device, queue, pl, "dependent", kk, n, runs, hasTS)
			if err != nil {
				return fmt.Errorf("dependent K=%d: %w", kk, err)
			}
			depG = g
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "dependent", kk, g, w, g*1000/float64(kk), "-")
			emit(csv, backendName, "dependent", kk, n, g, w, g*1000/float64(kk), 0)
		}
		if doIndep {
			g, w, err := bench(device, queue, pl, "independent", kk, n, runs, hasTS)
			if err != nil {
				return fmt.Errorf("independent K=%d: %w", kk, err)
			}
			indepG = g
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "independent", kk, g, w, g*1000/float64(kk), "-")
			emit(csv, backendName, "independent", kk, n, g, w, g*1000/float64(kk), 0)
		}
		if doDep && doIndep && depG > 0 {
			perBarr := (depG - indepG) * 1000 / float64(kk)
			perDisp := indepG * 1000 / float64(kk)
			fmt.Printf("%-12s %6d  %12s %12s  %14.3f %14.3f   <= isolated\n", "=>", kk, "", "", perDisp, perBarr)
			emit(csv, backendName, "isolated", kk, n, depG-indepG, 0, perDisp, perBarr)
		}
	}
	return nil
}

// bench builds the chain and returns median GPU ms (encoder-bracketed) and wall ms.
func bench(device *wgpu.Device, queue *wgpu.Queue, pl *wgpu.ComputePipeline, mode string, k, n, runs int, hasTS bool) (float64, float64, error) {
	size := uint64(n * 4)
	mkbuf := func() (*wgpu.Buffer, error) {
		return device.CreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageStorage})
	}
	mkbg := func(in, out *wgpu.Buffer) (*wgpu.BindGroup, error) {
		return device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout: pl.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: in, Size: in.GetSize()},
				{Binding: 1, Buffer: out, Size: out.GetSize()},
			},
		})
	}

	var bufs []*wgpu.Buffer
	bgs := make([]*wgpu.BindGroup, k)
	if mode == "dependent" {
		a, err := mkbuf()
		if err != nil {
			return 0, 0, err
		}
		b, err := mkbuf()
		if err != nil {
			return 0, 0, err
		}
		bufs = []*wgpu.Buffer{a, b}
		for i := 0; i < k; i++ {
			in, out := a, b
			if i%2 == 1 {
				in, out = b, a
			}
			bg, err := mkbg(in, out)
			if err != nil {
				return 0, 0, err
			}
			bgs[i] = bg
		}
	} else {
		shared, err := mkbuf()
		if err != nil {
			return 0, 0, err
		}
		bufs = append(bufs, shared)
		for i := 0; i < k; i++ {
			out, err := mkbuf()
			if err != nil {
				return 0, 0, err
			}
			bufs = append(bufs, out)
			bg, err := mkbg(shared, out)
			if err != nil {
				return 0, 0, err
			}
			bgs[i] = bg
		}
	}
	defer func() {
		for _, bg := range bgs {
			if bg != nil {
				bg.Release()
			}
		}
		for _, b := range bufs {
			if b != nil {
				b.Release()
			}
		}
	}()

	var qset *wgpu.QuerySet
	var resolve, stage *wgpu.Buffer
	if hasTS {
		var err error
		qset, err = device.CreateQuerySet(&wgpu.QuerySetDescriptor{Label: "ts", Type: wgpu.QueryTypeTimestamp, Count: 2})
		if err != nil {
			return 0, 0, err
		}
		defer qset.Release()
		resolve, err = device.CreateBuffer(&wgpu.BufferDescriptor{Size: 16, Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc})
		if err != nil {
			return 0, 0, err
		}
		defer resolve.Release()
		stage, err = device.CreateBuffer(&wgpu.BufferDescriptor{Size: 16, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
		if err != nil {
			return 0, 0, err
		}
		defer stage.Release()
	}

	groups := uint32((n + workgroupSize - 1) / workgroupSize)

	once := func() (float64, error) {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			return 0, err
		}
		if hasTS {
			if err := enc.WriteTimestamp(qset, 0); err != nil {
				return 0, err
			}
		}
		pass := enc.BeginComputePass(nil)
		pass.SetPipeline(pl)
		for i := 0; i < k; i++ {
			pass.SetBindGroup(0, bgs[i], nil)
			pass.DispatchWorkgroups(groups, 1, 1)
		}
		if err := pass.End(); err != nil {
			return 0, err
		}
		pass.Release()
		if hasTS {
			if err := enc.WriteTimestamp(qset, 1); err != nil {
				return 0, err
			}
			if err := enc.ResolveQuerySet(qset, 0, 2, resolve, 0); err != nil {
				return 0, err
			}
			if err := enc.CopyBufferToBuffer(resolve, 0, stage, 0, 16); err != nil {
				return 0, err
			}
		}
		cmd, err := enc.Finish(nil)
		if err != nil {
			return 0, err
		}
		queue.Submit(cmd)
		cmd.Release()
		enc.Release()
		device.Poll(true, nil)

		if !hasTS {
			return 0, nil
		}
		status := wgpu.BufferMapAsyncStatus(0)
		done := false
		if err := stage.MapAsync(wgpu.MapModeRead, 0, 16, func(s wgpu.BufferMapAsyncStatus) { status = s; done = true }); err != nil {
			return 0, err
		}
		device.Poll(true, nil)
		_ = done
		if status != wgpu.BufferMapAsyncStatusSuccess {
			return 0, fmt.Errorf("timestamp map failed: %v", status)
		}
		ts := wgpu.FromBytes[uint64](stage.GetMappedRange(0, 16))
		t0, t1 := ts[0], ts[1]
		stage.Unmap()
		if t1 < t0 {
			return 0, nil
		}
		return float64(t1-t0) * tsPeriod / 1e6, nil
	}

	if _, err := once(); err != nil { // warmup
		return 0, 0, err
	}
	gpu := make([]float64, runs)
	wall := make([]float64, runs)
	for r := 0; r < runs; r++ {
		start := time.Now()
		g, err := once()
		if err != nil {
			return 0, 0, err
		}
		wall[r] = float64(time.Since(start).Nanoseconds()) / 1e6
		gpu[r] = g
	}
	return median(gpu), median(wall), nil
}

func median(xs []float64) float64 {
	cp := append([]float64(nil), xs...)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j-1] > cp[j]; j-- {
			cp[j-1], cp[j] = cp[j], cp[j-1]
		}
	}
	if len(cp) == 0 {
		return 0
	}
	return cp[len(cp)/2]
}

func emit(on bool, backend, mode string, k, n int, gpuMS, wallMS, perDisp, perBarr float64) {
	if !on {
		return
	}
	fmt.Printf("csv,0xcoge0023,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f\n", backend, mode, k, n, gpuMS, wallMS, perDisp, perBarr)
}
