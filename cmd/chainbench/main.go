// Command chainbench is a self-contained synthetic benchmark for isolating the
// per-BARRIER and per-DISPATCH cost of a serially-dependent compute-dispatch
// chain on wgpu-native — the shape goinfer's ~535-dispatch decode chain takes.
//
// It builds two chains of K tiny compute dispatches:
//
//   - DEPENDENT: a ping-pong (A->B->A->...) where dispatch k reads the buffer
//     dispatch k-1 wrote and writes the buffer dispatch k+1 will read. Every
//     consecutive pair is a read-after-write hazard on a storage buffer, so
//     wgpu-hal MUST emit a memory/pipeline barrier between every dispatch. The
//     GPU cannot overlap them. Cost ~= K * (dispatch + barrier).
//
//   - INDEPENDENT (control): the same K dispatches, identical kernel and work
//     size, but each writes its OWN output buffer reading a shared constant
//     input. No read-after-write hazard, so no barriers are required and the
//     GPU is free to overlap the dispatches. Cost ~= K * dispatch.
//
// Subtracting isolates the per-barrier cost:
//
//	per_barrier = (dependent_gpu - independent_gpu) / K
//
// Work per dispatch is deliberately tiny (a few hundred elements, out[i]=in[i]+1)
// so the chain is latency-bound on barriers/launches, not ALU — mirroring the
// decode glue. Device-side time is measured with timestamp-query (falling back
// to wall clock); total wall clock is always reported.
//
// Usage:
//
//	go run ./cmd/chainbench                 # default: dependent vs independent, K=400
//	go run ./cmd/chainbench -k 535 -n 256   # mirror the real chain length
//	go run ./cmd/chainbench -backend vulkan # pin a backend (vulkan|gl|metal|d3d12)
//	go run ./cmd/chainbench -pass per       # one compute pass per dispatch
//	go run ./cmd/chainbench -submit per     # one queue submit (+poll) per dispatch
//	go run ./cmd/chainbench -ksweep 100,200,400,800 -csv   # bisect-friendly output
//
// The CSV columns (one row per (mode,K)) are stable across versions so the same
// binary linked against different libwgpu_native.a can be diffed directly:
//
//	csv,<wgpuVersionHex>,<backend>,<mode>,<K>,<n>,<pass>,<submit>,<gpu_ms>,<wall_ms>,<per_dispatch_us>,<per_barrier_us>
package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/townsendmerino/wgpu"
)

const workgroupSize = 64

// Trivial latency-bound kernel: out[i] = in[i] + 1. Tiny work on purpose — we
// are measuring launch+barrier latency, not throughput.
const chainWGSL = `
@group(0) @binding(0) var<storage, read>       inb:  array<u32>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&outb)) { return; }
    outb[i] = inb[i] + 1u;
}
`

type config struct {
	k       int
	n       int
	runs    int
	pass    string // "single" | "per"
	submit  string // "single" | "per"
	backend string // "" | "vulkan" | "gl" | "metal" | "d3d12" | "d3d11"
	mode    string // "both" | "dependent" | "independent"
	ksweep  string
	csv     bool
}

func main() {
	var cfg config
	flag.IntVar(&cfg.k, "k", 400, "chain length (number of serially-dependent dispatches)")
	flag.IntVar(&cfg.n, "n", 256, "elements per buffer (tiny = latency-bound)")
	flag.IntVar(&cfg.runs, "runs", 30, "timed repetitions (median reported)")
	flag.StringVar(&cfg.pass, "pass", "single", "dispatch grouping: single (all in one compute pass) | per (one pass each)")
	flag.StringVar(&cfg.submit, "submit", "single", "command-buffer grouping: single (one submit) | per (one submit + poll each)")
	flag.StringVar(&cfg.backend, "backend", "", "force backend: vulkan|gl|metal|d3d12|d3d11 (default: auto)")
	flag.StringVar(&cfg.mode, "mode", "both", "which chain: both|dependent|independent")
	flag.StringVar(&cfg.ksweep, "ksweep", "", "comma-separated K values to sweep (overrides -k), e.g. 100,200,400,800")
	flag.BoolVar(&cfg.csv, "csv", false, "emit machine-readable csv rows for cross-version diffing")
	flag.Parse()

	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "chainbench: FAIL:", err)
		os.Exit(1)
	}
}

func backendType(s string) wgpu.BackendType {
	switch strings.ToLower(s) {
	case "", "auto":
		return wgpu.BackendTypeUndefined
	case "vulkan", "vk":
		return wgpu.BackendTypeVulkan
	case "gl", "opengl":
		return wgpu.BackendTypeOpenGL
	case "gles", "opengles":
		return wgpu.BackendTypeOpenGLES
	case "metal", "mtl":
		return wgpu.BackendTypeMetal
	case "d3d12", "dx12":
		return wgpu.BackendTypeD3D12
	case "d3d11", "dx11":
		return wgpu.BackendTypeD3D11
	default:
		return wgpu.BackendTypeUndefined
	}
}

func run(cfg config) error {
	verHex := wgpu.GetVersion()
	fmt.Printf("wgpu-native version: 0x%08x\n", verHex)

	inst := wgpu.CreateInstance(nil)
	defer inst.Release()

	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
		BackendType:     backendType(cfg.backend),
	})
	if err != nil {
		return fmt.Errorf("request adapter (backend=%q): %w", cfg.backend, err)
	}
	defer adapter.Release()

	info := adapter.GetInfo()
	fmt.Printf("adapter: %s | backend=%s | type=%s | vendor=%s\n",
		info.Name, info.BackendType, info.AdapterType, info.VendorName)

	hasTimestamp := adapter.HasFeature(wgpu.FeatureNameTimestampQuery)
	fmt.Printf("timestamp-query: %v\n", hasTimestamp)

	lim := wgpu.DefaultLimits()
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

	sh, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label: "chain", WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: chainWGSL},
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

	ks, err := parseKs(cfg.ksweep, cfg.k)
	if err != nil {
		return err
	}
	backendName := info.BackendType.String()

	fmt.Printf("\nchain bench: n=%d elems/dispatch, runs=%d, pass=%s, submit=%s\n",
		cfg.n, cfg.runs, cfg.pass, cfg.submit)
	fmt.Printf("%-12s %6s  %12s %12s  %14s %14s\n", "mode", "K", "gpu(ms)", "wall(ms)", "per-disp(us)", "per-barr(us)")

	for _, k := range ks {
		var depGPU, indepGPU float64
		runOne := func(mode string) (gpuMS, wallMS float64, err error) {
			h := newHarness(device, queue, pl, cfg, k)
			defer h.release()
			if err := h.build(mode); err != nil {
				return 0, 0, err
			}
			return h.measure(hasTimestamp, queue)
		}

		doDep := cfg.mode == "both" || cfg.mode == "dependent"
		doIndep := cfg.mode == "both" || cfg.mode == "independent"

		if doDep {
			g, w, err := runOne("dependent")
			if err != nil {
				return fmt.Errorf("dependent K=%d: %w", k, err)
			}
			depGPU = g
			perDisp := g * 1000.0 / float64(k) // us per dispatch (incl barrier)
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "dependent", k, g, w, perDisp, "-")
			emitCSV(cfg.csv, verHex, backendName, "dependent", k, cfg, g, w, perDisp, 0, false)
		}
		if doIndep {
			g, w, err := runOne("independent")
			if err != nil {
				return fmt.Errorf("independent K=%d: %w", k, err)
			}
			indepGPU = g
			perDisp := g * 1000.0 / float64(k)
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "independent", k, g, w, perDisp, "-")
			emitCSV(cfg.csv, verHex, backendName, "independent", k, cfg, g, w, perDisp, 0, false)
		}
		if doDep && doIndep && depGPU > 0 {
			perBarrier := (depGPU - indepGPU) * 1000.0 / float64(k) // us
			perDispIndep := indepGPU * 1000.0 / float64(k)
			fmt.Printf("%-12s %6d  %12s %12s  %14.3f %14.3f   <= isolated\n",
				"=>", k, "", "", perDispIndep, perBarrier)
			emitCSV(cfg.csv, verHex, backendName, "isolated", k, cfg, depGPU-indepGPU, 0, perDispIndep, perBarrier, true)
		}
	}
	return nil
}

func parseKs(sweep string, single int) ([]int, error) {
	if strings.TrimSpace(sweep) == "" {
		return []int{single}, nil
	}
	var out []int
	for _, p := range strings.Split(sweep, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("bad -ksweep value %q: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func emitCSV(on bool, ver uint32, backend, mode string, k int, cfg config, gpuMS, wallMS, perDispUS, perBarrUS float64, isolated bool) {
	if !on {
		return
	}
	fmt.Printf("csv,0x%08x,%s,%s,%d,%d,%s,%s,%.4f,%.4f,%.4f,%.4f\n",
		ver, backend, mode, k, cfg.n, cfg.pass, cfg.submit, gpuMS, wallMS, perDispUS, perBarrUS)
}

// ---- harness ---------------------------------------------------------------

// harness owns the buffers/bind groups for one chain of K dispatches and knows
// how to encode + time the chain under the configured pass/submit granularity.
type harness struct {
	device *wgpu.Device
	queue  *wgpu.Queue
	pl     *wgpu.ComputePipeline
	cfg    config
	k      int

	bufs []*wgpu.Buffer
	bgs  []*wgpu.BindGroup // one per dispatch; bgs[i] is what dispatch i binds

	qset     *wgpu.QuerySet
	tsResolv *wgpu.Buffer
	tsStage  *wgpu.Buffer
}

func newHarness(d *wgpu.Device, q *wgpu.Queue, pl *wgpu.ComputePipeline, cfg config, k int) *harness {
	return &harness{device: d, queue: q, pl: pl, cfg: cfg, k: k}
}

// build allocates buffers and bind groups for the requested chain shape.
//
//	dependent:   2 ping-pong buffers; bgs alternate (A->B, B->A, ...). Every
//	             dispatch reads what the previous wrote => RAW hazard each step.
//	independent: 1 shared input + K disjoint output buffers; bgs[i] writes its
//	             own buffer => no hazard, dispatches may overlap.
func (h *harness) build(mode string) error {
	sizeBytes := uint64(h.cfg.n * 4)
	mk := func(label string, usage wgpu.BufferUsage) (*wgpu.Buffer, error) {
		return h.device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: sizeBytes, Usage: usage})
	}
	mkbg := func(in, out *wgpu.Buffer) (*wgpu.BindGroup, error) {
		return h.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
			Layout: h.pl.GetBindGroupLayout(0),
			Entries: []wgpu.BindGroupEntry{
				{Binding: 0, Buffer: in, Size: in.GetSize()},
				{Binding: 1, Buffer: out, Size: out.GetSize()},
			},
		})
	}

	switch mode {
	case "dependent":
		a, err := mk("ping-A", wgpu.BufferUsageStorage)
		if err != nil {
			return err
		}
		b, err := mk("ping-B", wgpu.BufferUsageStorage)
		if err != nil {
			return err
		}
		h.bufs = []*wgpu.Buffer{a, b}
		// dispatch i reads bufs[i%2], writes bufs[(i+1)%2]: a->b->a->b...
		h.bgs = make([]*wgpu.BindGroup, h.k)
		for i := 0; i < h.k; i++ {
			in, out := a, b
			if i%2 == 1 {
				in, out = b, a
			}
			bg, err := mkbg(in, out)
			if err != nil {
				return err
			}
			h.bgs[i] = bg
		}
	case "independent":
		shared, err := mk("shared-in", wgpu.BufferUsageStorage)
		if err != nil {
			return err
		}
		h.bufs = append(h.bufs, shared)
		h.bgs = make([]*wgpu.BindGroup, h.k)
		for i := 0; i < h.k; i++ {
			out, err := mk("out-"+strconv.Itoa(i), wgpu.BufferUsageStorage)
			if err != nil {
				return err
			}
			h.bufs = append(h.bufs, out)
			bg, err := mkbg(shared, out)
			if err != nil {
				return err
			}
			h.bgs[i] = bg
		}
	default:
		return fmt.Errorf("unknown mode %q", mode)
	}

	// Timestamp plumbing (2 queries: start + end of the whole chain).
	qset, err := h.device.CreateQuerySet(&wgpu.QuerySetDescriptor{Label: "ts", Type: wgpu.QueryTypeTimestamp, Count: 2})
	if err == nil {
		h.qset = qset
		h.tsResolv, _ = h.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "tsR", Size: 16, Usage: wgpu.BufferUsageQueryResolve | wgpu.BufferUsageCopySrc})
		h.tsStage, _ = h.device.CreateBuffer(&wgpu.BufferDescriptor{Label: "tsS", Size: 16, Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst})
	}
	return nil
}

func (h *harness) release() {
	for _, bg := range h.bgs {
		if bg != nil {
			bg.Release()
		}
	}
	for _, b := range h.bufs {
		if b != nil {
			b.Release()
		}
	}
	if h.qset != nil {
		h.qset.Release()
	}
	if h.tsResolv != nil {
		h.tsResolv.Release()
	}
	if h.tsStage != nil {
		h.tsStage.Release()
	}
}

// groups is the workgroup count to cover n elements.
func (h *harness) groups() uint32 {
	return uint32((h.cfg.n + workgroupSize - 1) / workgroupSize)
}

// measure runs warmup + cfg.runs timed iterations and returns median GPU(ms)
// (timestamp-based when available, else 0) and median wall(ms).
func (h *harness) measure(hasTimestamp bool, queue *wgpu.Queue) (float64, float64, error) {
	period := 1.0
	if hasTimestamp && h.qset != nil {
		period = float64(queue.GetTimestampPeriod())
	} else {
		hasTimestamp = false
	}

	// warmup
	if _, err := h.encodeAndRun(false, period); err != nil {
		return 0, 0, err
	}

	gpu := make([]float64, 0, h.cfg.runs)
	wall := make([]float64, 0, h.cfg.runs)
	for r := 0; r < h.cfg.runs; r++ {
		start := time.Now()
		g, err := h.encodeAndRun(hasTimestamp, period)
		w := float64(time.Since(start).Nanoseconds()) / 1e6
		if err != nil {
			return 0, 0, err
		}
		gpu = append(gpu, g)
		wall = append(wall, w)
	}
	return median(gpu), median(wall), nil
}

// encodeAndRun encodes the K-dispatch chain per the pass/submit config, submits,
// polls to completion, and returns device-side ms (0 if no timestamps). The
// whole-chain timestamps are only valid for pass=single,submit=single; in the
// other granularities we report 0 GPU ms and rely on wall clock.
func (h *harness) encodeAndRun(useTS bool, period float64) (float64, error) {
	singlePass := h.cfg.pass == "single"
	singleSubmit := h.cfg.submit == "single"
	canTS := useTS && h.qset != nil && singlePass && singleSubmit

	if singleSubmit {
		enc, err := h.device.CreateCommandEncoder(nil)
		if err != nil {
			return 0, err
		}
		defer enc.Release()
		if err := h.encodeChain(enc, singlePass, canTS); err != nil {
			return 0, err
		}
		if canTS {
			enc.ResolveQuerySet(h.qset, 0, 2, h.tsResolv, 0)
			if err := enc.CopyBufferToBuffer(h.tsResolv, 0, h.tsStage, 0, 16); err != nil {
				return 0, err
			}
		}
		cmd, err := enc.Finish(nil)
		if err != nil {
			return 0, err
		}
		h.queue.Submit(cmd)
		cmd.Release()
		h.device.Poll(true, nil)
	} else {
		// One encoder+submit+poll per dispatch.
		for i := 0; i < h.k; i++ {
			enc, err := h.device.CreateCommandEncoder(nil)
			if err != nil {
				return 0, err
			}
			pass := enc.BeginComputePass(nil)
			pass.SetPipeline(h.pl)
			pass.SetBindGroup(0, h.bgs[i], nil)
			pass.DispatchWorkgroups(h.groups(), 1, 1)
			if err := pass.End(); err != nil {
				return 0, err
			}
			pass.Release()
			cmd, err := enc.Finish(nil)
			if err != nil {
				return 0, err
			}
			h.queue.Submit(cmd)
			cmd.Release()
			enc.Release()
			h.device.Poll(true, nil)
		}
	}

	if !canTS {
		return 0, nil
	}
	return h.readTS(period)
}

// encodeChain records the K dispatches into enc, either as one big compute pass
// (singlePass) or one pass per dispatch. withTS attaches whole-chain timestamps
// to the first/last pass (only meaningful for singlePass).
func (h *harness) encodeChain(enc *wgpu.CommandEncoder, singlePass, withTS bool) error {
	if singlePass {
		var pd *wgpu.ComputePassDescriptor
		if withTS {
			pd = &wgpu.ComputePassDescriptor{TimestampWrites: &wgpu.ComputePassTimestampWrites{
				QuerySet: h.qset, BeginningOfPassWriteIndex: 0, EndOfPassWriteIndex: 1,
			}}
		}
		pass := enc.BeginComputePass(pd)
		pass.SetPipeline(h.pl)
		for i := 0; i < h.k; i++ {
			pass.SetBindGroup(0, h.bgs[i], nil)
			pass.DispatchWorkgroups(h.groups(), 1, 1)
		}
		if err := pass.End(); err != nil {
			return err
		}
		pass.Release()
		return nil
	}
	// One pass per dispatch within a single command buffer. Timestamps span the
	// first and last pass when requested.
	for i := 0; i < h.k; i++ {
		var pd *wgpu.ComputePassDescriptor
		if withTS && (i == 0 || i == h.k-1) {
			tw := &wgpu.ComputePassTimestampWrites{QuerySet: h.qset,
				BeginningOfPassWriteIndex: wgpu.QuerySetIndexUndefined,
				EndOfPassWriteIndex:       wgpu.QuerySetIndexUndefined}
			if i == 0 {
				tw.BeginningOfPassWriteIndex = 0
			}
			if i == h.k-1 {
				tw.EndOfPassWriteIndex = 1
			}
			pd = &wgpu.ComputePassDescriptor{TimestampWrites: tw}
		}
		pass := enc.BeginComputePass(pd)
		pass.SetPipeline(h.pl)
		pass.SetBindGroup(0, h.bgs[i], nil)
		pass.DispatchWorkgroups(h.groups(), 1, 1)
		if err := pass.End(); err != nil {
			return err
		}
		pass.Release()
	}
	return nil
}

func (h *harness) readTS(period float64) (float64, error) {
	status := wgpu.BufferMapAsyncStatusUnknown
	if err := h.tsStage.MapAsync(wgpu.MapModeRead, 0, 16, func(s wgpu.BufferMapAsyncStatus) { status = s }); err != nil {
		return 0, err
	}
	h.device.Poll(true, nil)
	if status != wgpu.BufferMapAsyncStatusSuccess {
		return 0, fmt.Errorf("timestamp map failed: %v", status)
	}
	ts := wgpu.FromBytes[uint64](h.tsStage.GetMappedRange(0, 16))
	t0, t1 := ts[0], ts[1]
	if err := h.tsStage.Unmap(); err != nil {
		return 0, err
	}
	if t1 < t0 {
		return 0, nil
	}
	return float64(t1-t0) * period / 1e6, nil
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	// simple insertion sort (small n)
	cp := append([]float64(nil), xs...)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j-1] > cp[j]; j-- {
			cp[j-1], cp[j] = cp[j], cp[j-1]
		}
	}
	return cp[len(cp)/2]
}
