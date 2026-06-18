// Command chainprobe is a SELF-CONTAINED, version-portable variant of the
// chainbench experiment used for the wgpu-native version bisect (STEP 2) and as
// the minimal standalone repro for an upstream gfx-rs/wgpu issue (STEP 4).
//
// Unlike cmd/chainbench it does NOT import the (v29-tuned) wgpu binding package.
// It talks to libwgpu_native directly through cgo using only the subset of the
// C API that is byte-for-byte stable across v25.0.2.x, v27.0.4.x and v29.0.0.0:
// instance/adapter/device/queue, shader+compute pipeline, storage buffers,
// auto bind-group layouts, a single compute pass, and timestamp queries. It
// passes requiredLimits=NULL and never touches the fields that churned between
// versions (subgroup sizes, immediates/push-constant limits), so the SAME
// source compiles against every release by only swapping headers+lib.
//
// The two ABI deltas in this subset are handled by compile flags, set per
// release by scripts/bisect.sh:
//
//	-DWGPU_TS_LEGACY     v25 names the struct WGPUComputePassTimestampWrites;
//	                     v27/v29 renamed it to WGPUPassTimestampWrites.
//	-DWGPU_NO_TS_PERIOD  v25 lacks wgpuQueueGetTimestampPeriod; pass the GPU's
//	                     (hardware-constant) period via -DTS_PERIOD=<float>.
//
// It measures exactly what chainbench does:
//
//	dependent chain (ping-pong, read-after-write every dispatch => one barrier
//	  per dispatch)      ~= K * (dispatch + barrier)
//	independent chain (disjoint outputs, no hazard => GPU may overlap)
//	                     ~= K * dispatch
//	per_barrier = (dependent_gpu - independent_gpu) / K
//
// Output is a stable CSV line per (mode,K) so runs against different libs diff
// directly:
//
//	csv,<verHex>,<backend>,<mode>,<K>,<n>,<gpu_ms>,<wall_ms>,<per_dispatch_us>,<per_barrier_us>
//
// Usage:
//
//	go build ./cmd/chainprobe && ./chainprobe -backend vulkan -ksweep 100,200,400
package main

/*
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/../../lib/linux/amd64 -lwgpu_native
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/../../lib/linux/arm64 -lwgpu_native
#cgo linux        LDFLAGS: -lm -ldl -lpthread
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/../../lib/darwin/amd64 -lwgpu_native
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/../../lib/darwin/arm64 -lwgpu_native
#cgo darwin       LDFLAGS: -framework QuartzCore -framework Metal -framework Foundation

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../lib/webgpu.h"
#include "../../lib/wgpu.h"

// ---- ABI-delta shims (see package doc) ------------------------------------
#if defined(WGPU_TS_LEGACY)
typedef WGPUComputePassTimestampWrites cpPassTS;
#else
typedef WGPUPassTimestampWrites cpPassTS;
#endif

#if defined(WGPU_NO_TS_PERIOD)
#ifndef TS_PERIOD
#define TS_PERIOD 1.0f
#endif
static float cp_period(WGPUQueue q){ (void)q; return (float)TS_PERIOD; }
#else
static float cp_period(WGPUQueue q){ return wgpuQueueGetTimestampPeriod(q); }
#endif

static WGPUStringView sv(const char* s){ WGPUStringView v; v.data=s; v.length=strlen(s); return v; }

static double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e3 + (double)ts.tv_nsec/1e6;
}

// ---- sync wrappers over the async future API (stable across v25..v29) ------
typedef struct { uint32_t status; WGPUAdapter adapter; int done; } adRes;
static void adCB(WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView m, void* u1, void* u2){
    (void)m;(void)u2; adRes* r=(adRes*)u1; r->status=(uint32_t)s; r->adapter=a; r->done=1;
}
typedef struct { uint32_t status; WGPUDevice device; int done; } dvRes;
static void dvCB(WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView m, void* u1, void* u2){
    (void)m;(void)u2; dvRes* r=(dvRes*)u1; r->status=(uint32_t)s; r->device=d; r->done=1;
}
typedef struct { uint32_t status; int done; } mapRes;
static void mapCB(WGPUMapAsyncStatus s, WGPUStringView m, void* u1, void* u2){
    (void)m;(void)u2; mapRes* r=(mapRes*)u1; r->status=(uint32_t)s; r->done=1;
}

// ---- globals owned by cp_setup / cp_teardown ------------------------------
static WGPUInstance        g_inst;
static WGPUAdapter         g_adapter;
static WGPUDevice          g_device;
static WGPUQueue           g_queue;
static WGPUComputePipeline g_pipe;
static WGPUBindGroupLayout g_bgl;
static int                 g_hasTS;
static float               g_period;

static const char* KERNEL =
"@group(0) @binding(0) var<storage, read>       inb:  array<u32>;\n"
"@group(0) @binding(1) var<storage, read_write> outb: array<u32>;\n"
"@compute @workgroup_size(64)\n"
"fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
"    let i = gid.x;\n"
"    if (i >= arrayLength(&outb)) { return; }\n"
"    outb[i] = inb[i] + 1u;\n"
"}\n";

// cp_setup brings up instance/adapter/device/queue/pipeline for one backend.
// backend is a WGPUBackendType (0 = auto). Returns 0 on success.
static int cp_setup(uint32_t backend, char* nameOut, int nameCap, uint32_t* backendOut, int* hasTSout, uint32_t* verOut){
    *verOut = wgpuGetVersion();
    g_inst = wgpuCreateInstance(NULL);
    if(!g_inst) return 1;

    WGPURequestAdapterOptions opt; memset(&opt,0,sizeof(opt));
    opt.powerPreference = WGPUPowerPreference_HighPerformance;
    opt.backendType = (WGPUBackendType)backend;
    adRes ar; ar.status=0; ar.adapter=NULL; ar.done=0;
    WGPURequestAdapterCallbackInfo aci; memset(&aci,0,sizeof(aci));
    aci.mode = WGPUCallbackMode_AllowProcessEvents; aci.callback = adCB; aci.userdata1=&ar;
    wgpuInstanceRequestAdapter(g_inst, &opt, aci);
    for(int g=0; !ar.done && g<1000000; ++g) wgpuInstanceProcessEvents(g_inst);
    if(!ar.adapter) return 2;
    g_adapter = ar.adapter;

    WGPUAdapterInfo info; memset(&info,0,sizeof(info));
    wgpuAdapterGetInfo(g_adapter, &info);
    if(info.device.data && nameOut){
        size_t n = info.device.length < (size_t)(nameCap-1) ? info.device.length : (size_t)(nameCap-1);
        memcpy(nameOut, info.device.data, n); nameOut[n]=0;
    }
    *backendOut = (uint32_t)info.backendType;
    wgpuAdapterInfoFreeMembers(info);

    g_hasTS = wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery) ? 1 : 0;
    *hasTSout = g_hasTS;

    WGPUDeviceDescriptor dd; memset(&dd,0,sizeof(dd));
    WGPUFeatureName feats[1] = { WGPUFeatureName_TimestampQuery };
    if(g_hasTS){ dd.requiredFeatures = feats; dd.requiredFeatureCount = 1; }
    dvRes dr; dr.status=0; dr.device=NULL; dr.done=0;
    WGPURequestDeviceCallbackInfo dci; memset(&dci,0,sizeof(dci));
    dci.mode = WGPUCallbackMode_AllowProcessEvents; dci.callback = dvCB; dci.userdata1=&dr;
    wgpuAdapterRequestDevice(g_adapter, &dd, dci);
    for(int g=0; !dr.done && g<1000000; ++g) wgpuInstanceProcessEvents(g_inst);
    if(!dr.device) return 3;
    g_device = dr.device;
    g_queue  = wgpuDeviceGetQueue(g_device);
    g_period = g_hasTS ? cp_period(g_queue) : 1.0f;

    WGPUShaderSourceWGSL wgsl; memset(&wgsl,0,sizeof(wgsl));
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = sv(KERNEL);
    WGPUShaderModuleDescriptor smd; memset(&smd,0,sizeof(smd));
    smd.nextInChain = &wgsl.chain;
    WGPUShaderModule sh = wgpuDeviceCreateShaderModule(g_device, &smd);
    if(!sh) return 4;

    WGPUComputePipelineDescriptor cpd; memset(&cpd,0,sizeof(cpd));
    cpd.compute.module = sh;
    cpd.compute.entryPoint = sv("main");
    g_pipe = wgpuDeviceCreateComputePipeline(g_device, &cpd);
    wgpuShaderModuleRelease(sh);
    if(!g_pipe) return 5;
    g_bgl = wgpuComputePipelineGetBindGroupLayout(g_pipe, 0);
    return 0;
}

static WGPUBuffer mkbuf(uint64_t size){
    WGPUBufferDescriptor bd; memset(&bd,0,sizeof(bd));
    bd.size = size; bd.usage = WGPUBufferUsage_Storage;
    return wgpuDeviceCreateBuffer(g_device, &bd);
}
static WGPUBindGroup mkbg(WGPUBuffer in, WGPUBuffer out, uint64_t size){
    WGPUBindGroupEntry e[2]; memset(e,0,sizeof(e));
    e[0].binding=0; e[0].buffer=in;  e[0].size=size;
    e[1].binding=1; e[1].buffer=out; e[1].size=size;
    WGPUBindGroupDescriptor bgd; memset(&bgd,0,sizeof(bgd));
    bgd.layout=g_bgl; bgd.entryCount=2; bgd.entries=e;
    return wgpuDeviceCreateBindGroup(g_device, &bgd);
}

// cp_bench runs one chain of K dispatches, n elems each, `runs` timed reps after
// a warmup, and writes the MEDIAN device-side ms (0 if no timestamps) and wall
// ms. mode 0 = dependent (ping-pong RAW), 1 = independent (disjoint outputs).
static int cp_bench(int mode, int K, int n, int runs, double* gpuMsOut, double* wallMsOut){
    uint64_t size = (uint64_t)n*4;
    uint32_t groups = (uint32_t)((n + 63)/64);

    // Build buffers + per-dispatch bind groups.
    WGPUBuffer* bufs; int nbufs;
    WGPUBindGroup* bgs = (WGPUBindGroup*)calloc(K, sizeof(WGPUBindGroup));
    if(mode==0){
        nbufs=2; bufs=(WGPUBuffer*)calloc(2,sizeof(WGPUBuffer));
        bufs[0]=mkbuf(size); bufs[1]=mkbuf(size);
        for(int i=0;i<K;i++){
            WGPUBuffer in = bufs[i&1], out = bufs[(i+1)&1];
            bgs[i]=mkbg(in,out,size);
        }
    } else {
        nbufs=K+1; bufs=(WGPUBuffer*)calloc(nbufs,sizeof(WGPUBuffer));
        bufs[0]=mkbuf(size); // shared input
        for(int i=0;i<K;i++){ bufs[i+1]=mkbuf(size); bgs[i]=mkbg(bufs[0],bufs[i+1],size); }
    }

    WGPUQuerySet qset=NULL; WGPUBuffer tsResolve=NULL, tsStage=NULL;
    if(g_hasTS){
        WGPUQuerySetDescriptor qd; memset(&qd,0,sizeof(qd));
        qd.type=WGPUQueryType_Timestamp; qd.count=2;
        qset=wgpuDeviceCreateQuerySet(g_device,&qd);
        WGPUBufferDescriptor rd; memset(&rd,0,sizeof(rd));
        rd.size=16; rd.usage=WGPUBufferUsage_QueryResolve|WGPUBufferUsage_CopySrc;
        tsResolve=wgpuDeviceCreateBuffer(g_device,&rd);
        WGPUBufferDescriptor sd; memset(&sd,0,sizeof(sd));
        sd.size=16; sd.usage=WGPUBufferUsage_MapRead|WGPUBufferUsage_CopyDst;
        tsStage=wgpuDeviceCreateBuffer(g_device,&sd);
    }

    // one timed iteration; returns gpu_ms (0 if no TS)
    #define RUN_ONCE(useTS, gpuOut) do { \
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, NULL); \
        cpPassTS tw; memset(&tw,0,sizeof(tw)); \
        tw.querySet=qset; tw.beginningOfPassWriteIndex=0; tw.endOfPassWriteIndex=1; \
        WGPUComputePassDescriptor pd; memset(&pd,0,sizeof(pd)); \
        if(useTS) pd.timestampWrites=&tw; \
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pd); \
        wgpuComputePassEncoderSetPipeline(pass, g_pipe); \
        for(int i=0;i<K;i++){ \
            wgpuComputePassEncoderSetBindGroup(pass,0,bgs[i],0,NULL); \
            wgpuComputePassEncoderDispatchWorkgroups(pass,groups,1,1); \
        } \
        wgpuComputePassEncoderEnd(pass); \
        wgpuComputePassEncoderRelease(pass); \
        if(useTS){ \
            wgpuCommandEncoderResolveQuerySet(enc, qset, 0, 2, tsResolve, 0); \
            wgpuCommandEncoderCopyBufferToBuffer(enc, tsResolve, 0, tsStage, 0, 16); \
        } \
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL); \
        wgpuQueueSubmit(g_queue, 1, &cmd); \
        wgpuCommandBufferRelease(cmd); \
        wgpuCommandEncoderRelease(enc); \
        wgpuDevicePoll(g_device, 1, NULL); \
        gpuOut = 0.0; \
        if(useTS){ \
            mapRes mr; mr.status=0; mr.done=0; \
            WGPUBufferMapCallbackInfo ci; memset(&ci,0,sizeof(ci)); \
            ci.mode=WGPUCallbackMode_AllowProcessEvents; ci.callback=mapCB; ci.userdata1=&mr; \
            wgpuBufferMapAsync(tsStage, WGPUMapMode_Read, 0, 16, ci); \
            wgpuDevicePoll(g_device, 1, NULL); \
            const uint64_t* ts = (const uint64_t*)wgpuBufferGetMappedRange(tsStage, 0, 16); \
            if(ts && ts[1]>=ts[0]) gpuOut = (double)(ts[1]-ts[0]) * (double)g_period / 1e6; \
            wgpuBufferUnmap(tsStage); \
        } \
    } while(0)

    double dummy; (void)dummy;
    double w0; RUN_ONCE(g_hasTS, dummy); (void)w0; // warmup

    double* gpu = (double*)calloc(runs,sizeof(double));
    double* wal = (double*)calloc(runs,sizeof(double));
    for(int r=0;r<runs;r++){
        double t0=now_ms(); double g;
        RUN_ONCE(g_hasTS, g);
        double t1=now_ms();
        gpu[r]=g; wal[r]=t1-t0;
    }
    // median (insertion sort, small runs)
    for(int i=1;i<runs;i++){ for(int j=i;j>0&&gpu[j-1]>gpu[j];j--){double t=gpu[j-1];gpu[j-1]=gpu[j];gpu[j]=t;} }
    for(int i=1;i<runs;i++){ for(int j=i;j>0&&wal[j-1]>wal[j];j--){double t=wal[j-1];wal[j-1]=wal[j];wal[j]=t;} }
    *gpuMsOut = gpu[runs/2];
    *wallMsOut = wal[runs/2];
    free(gpu); free(wal);

    // cleanup
    for(int i=0;i<K;i++) if(bgs[i]) wgpuBindGroupRelease(bgs[i]);
    for(int i=0;i<nbufs;i++) if(bufs[i]) wgpuBufferRelease(bufs[i]);
    free(bgs); free(bufs);
    if(qset) wgpuQuerySetRelease(qset);
    if(tsResolve) wgpuBufferRelease(tsResolve);
    if(tsStage) wgpuBufferRelease(tsStage);
    return 0;
}

static void cp_teardown(void){
    if(g_bgl) wgpuBindGroupLayoutRelease(g_bgl);
    if(g_pipe) wgpuComputePipelineRelease(g_pipe);
    if(g_queue) wgpuQueueRelease(g_queue);
    if(g_device) wgpuDeviceRelease(g_device);
    if(g_adapter) wgpuAdapterRelease(g_adapter);
    if(g_inst) wgpuInstanceRelease(g_inst);
}
*/
import "C"

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unsafe"
)

func backendCode(s string) C.uint32_t {
	switch strings.ToLower(s) {
	case "vulkan", "vk":
		return C.WGPUBackendType_Vulkan
	case "gl", "opengl":
		return C.WGPUBackendType_OpenGL
	case "gles", "opengles":
		return C.WGPUBackendType_OpenGLES
	case "metal", "mtl":
		return C.WGPUBackendType_Metal
	case "d3d12", "dx12":
		return C.WGPUBackendType_D3D12
	default:
		return C.WGPUBackendType_Undefined
	}
}

func backendName(c uint32) string {
	switch C.WGPUBackendType(c) {
	case C.WGPUBackendType_Vulkan:
		return "Vulkan"
	case C.WGPUBackendType_Metal:
		return "Metal"
	case C.WGPUBackendType_OpenGL:
		return "OpenGL"
	case C.WGPUBackendType_OpenGLES:
		return "OpenGLES"
	case C.WGPUBackendType_D3D12:
		return "D3D12"
	default:
		return "Other"
	}
}

func main() {
	backend := flag.String("backend", "", "force backend: vulkan|gl|metal|d3d12 (default auto)")
	k := flag.Int("k", 400, "chain length")
	n := flag.Int("n", 256, "elements per buffer")
	runs := flag.Int("runs", 30, "timed reps (median)")
	ksweep := flag.String("ksweep", "", "comma-separated K values (overrides -k)")
	mode := flag.String("mode", "both", "both|dependent|independent")
	csv := flag.Bool("csv", false, "emit csv rows")
	flag.Parse()

	var name [256]C.char
	var bk C.uint32_t
	var hasTS C.int
	var ver C.uint32_t
	if rc := C.cp_setup(backendCode(*backend), &name[0], 256, &bk, &hasTS, &ver); rc != 0 {
		fmt.Fprintf(os.Stderr, "chainprobe: cp_setup failed rc=%d (backend=%q)\n", int(rc), *backend)
		os.Exit(1)
	}
	defer C.cp_teardown()

	bn := backendName(uint32(bk))
	fmt.Printf("wgpu-native version: 0x%08x\n", uint32(ver))
	fmt.Printf("adapter: %s | backend=%s | timestamps=%v\n", C.GoString(&name[0]), bn, hasTS != 0)

	ks := []int{*k}
	if strings.TrimSpace(*ksweep) != "" {
		ks = ks[:0]
		for _, p := range strings.Split(*ksweep, ",") {
			if p = strings.TrimSpace(p); p == "" {
				continue
			}
			v, err := strconv.Atoi(p)
			if err != nil {
				fmt.Fprintf(os.Stderr, "bad -ksweep %q: %v\n", p, err)
				os.Exit(1)
			}
			ks = append(ks, v)
		}
	}

	doDep := *mode == "both" || *mode == "dependent"
	doIndep := *mode == "both" || *mode == "independent"

	fmt.Printf("\nn=%d elems/dispatch, runs=%d\n", *n, *runs)
	fmt.Printf("%-12s %6s  %12s %12s  %14s %14s\n", "mode", "K", "gpu(ms)", "wall(ms)", "per-disp(us)", "per-barr(us)")

	run := func(m int, kk int) (float64, float64) {
		var g, w C.double
		C.cp_bench(C.int(m), C.int(kk), C.int(*n), C.int(*runs), &g, &w)
		return float64(g), float64(w)
	}
	_ = unsafe.Pointer(nil)

	for _, kk := range ks {
		var depG, indepG float64
		if doDep {
			g, w := run(0, kk)
			depG = g
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "dependent", kk, g, w, g*1000/float64(kk), "-")
			emit(*csv, uint32(ver), bn, "dependent", kk, *n, g, w, g*1000/float64(kk), 0)
		}
		if doIndep {
			g, w := run(1, kk)
			indepG = g
			fmt.Printf("%-12s %6d  %12.3f %12.3f  %14.3f %14s\n", "independent", kk, g, w, g*1000/float64(kk), "-")
			emit(*csv, uint32(ver), bn, "independent", kk, *n, g, w, g*1000/float64(kk), 0)
		}
		if doDep && doIndep && depG > 0 {
			perBarr := (depG - indepG) * 1000 / float64(kk)
			perDisp := indepG * 1000 / float64(kk)
			fmt.Printf("%-12s %6d  %12s %12s  %14.3f %14.3f   <= isolated\n", "=>", kk, "", "", perDisp, perBarr)
			emit(*csv, uint32(ver), bn, "isolated", kk, *n, depG-indepG, 0, perDisp, perBarr)
		}
	}
}

func emit(on bool, ver uint32, backend, mode string, k, n int, gpuMS, wallMS, perDisp, perBarr float64) {
	if !on {
		return
	}
	fmt.Printf("csv,0x%08x,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f\n", ver, backend, mode, k, n, gpuMS, wallMS, perDisp, perBarr)
}
