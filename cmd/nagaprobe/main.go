// Command nagaprobe is the decisive naga-codegen A/B (STEP "follow-up 3"):
// does v29's naga generate slower SPIR-V than v22-era naga for goinfer's real
// glue kernels? It isolates the COMPILER from the runtime by running two
// precompiled SPIR-V blobs (produced offline by naga-cli 22.0.0 and 29.0.0)
// on the SAME v29 wgpu-native runtime via SpirvShaderPassthrough
// (wgpuDeviceCreateShaderModuleSpirV) — the runtime's own naga never runs, so
// the driver receives exactly the bytes each naga version emitted.
//
// For each fixture (real goinfer glue WGSL under fixtures/) it runs the same
// serially-dependent chain as chainprobe — K dispatches, one shared read-write
// storage buffer reused so each dispatch is a RAW hazard on the previous (1
// barrier/dispatch), tiny per-dispatch work — and reports device-side
// per-dispatch GPU µs (timestamp queries) for:
//   (a) SPIR-V from naga 22.0.0  (-spv ...22.spv)
//   (b) SPIR-V from naga 29.0.0  (-spv ...29.spv)
//   (c) the WGSL compiled by the v29 RUNTIME's own naga (-wgsl ...)  [sanity:
//       should match (b), confirming naga-cli 29 ≈ the runtime's naga 29.0.1]
//
// switch of interest = per-dispatch(a) vs per-dispatch(b): if (a) is meaningfully
// faster, v29 naga regressed codegen for that kernel.
//
// Explicit bind-group layouts are built per fixture (no reflection), so the
// passthrough path is identical for every blob.
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
#include <stdio.h>
#include <time.h>
#include "../../lib/webgpu.h"
#include "../../lib/wgpu.h"

static WGPUStringView sv(const char* s){ WGPUStringView v; v.data=s; v.length=strlen(s); return v; }

typedef struct { uint32_t status; WGPUAdapter adapter; int done; } adRes;
static void adCB(WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView m, void* u1, void* u2){ (void)m;(void)u2; adRes* r=(adRes*)u1; r->status=(uint32_t)s; r->adapter=a; r->done=1; }
typedef struct { uint32_t status; WGPUDevice device; int done; } dvRes;
static void dvCB(WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView m, void* u1, void* u2){ (void)m;(void)u2; dvRes* r=(dvRes*)u1; r->status=(uint32_t)s; r->device=d; r->done=1; }
typedef struct { uint32_t status; int done; } mapRes;
static void mapCB(WGPUMapAsyncStatus s, WGPUStringView m, void* u1, void* u2){ (void)m;(void)u2; mapRes* r=(mapRes*)u1; r->status=(uint32_t)s; r->done=1; }

#define NP_MAX_PIPE 16
static WGPUInstance        g_inst;
static WGPUAdapter         g_adapter;
static WGPUDevice          g_device;
static WGPUQueue           g_queue;
static int                 g_hasTS;
static float               g_period;
static WGPUComputePipeline g_pipes[NP_MAX_PIPE];
static WGPUBindGroupLayout g_bgls[NP_MAX_PIPE];
static WGPUPipelineLayout  g_plls[NP_MAX_PIPE];
static int                 g_npipes;

static int np_init(uint32_t backend, char* nameOut, int nameCap, int* hasPassOut, int* hasTSout, uint32_t* verOut){
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
    if(info.device.data && nameOut){ size_t n=info.device.length<(size_t)(nameCap-1)?info.device.length:(size_t)(nameCap-1); memcpy(nameOut,info.device.data,n); nameOut[n]=0; }
    wgpuAdapterInfoFreeMembers(info);

    { // diagnostic: list adapter features (find passthrough's advertised value)
        WGPUSupportedFeatures sf; memset(&sf,0,sizeof(sf));
        wgpuAdapterGetFeatures(g_adapter, &sf);
        fprintf(stderr, "adapter features (%zu):", sf.featureCount);
        for(size_t i=0;i<sf.featureCount;i++) fprintf(stderr, " 0x%08x", (unsigned)sf.features[i]);
        fprintf(stderr, "\n");
        wgpuSupportedFeaturesFreeMembers(sf);
    }
    g_hasTS = wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery)?1:0;

    // SpirvShaderPassthrough is a wgpu-native native feature that HasFeature does
    // not reliably enumerate; the real test is whether the device accepts it.
    // Request it (plus timestamps); if device creation fails, retry without and
    // report passthrough as unavailable.
    WGPUDevice dev = NULL; int hasPass = 0;
    for(int attempt=0; attempt<2 && !dev; attempt++){
        WGPUFeatureName feats[2]; int nf=0;
        if(g_hasTS) feats[nf++] = WGPUFeatureName_TimestampQuery;
        if(attempt==0) feats[nf++] = (WGPUFeatureName)WGPUNativeFeature_SpirvShaderPassthrough;
        WGPUDeviceDescriptor dd; memset(&dd,0,sizeof(dd));
        dd.requiredFeatures = feats; dd.requiredFeatureCount = (size_t)nf;
        dvRes dr; dr.status=0; dr.device=NULL; dr.done=0;
        WGPURequestDeviceCallbackInfo dci; memset(&dci,0,sizeof(dci));
        dci.mode = WGPUCallbackMode_AllowProcessEvents; dci.callback = dvCB; dci.userdata1=&dr;
        wgpuAdapterRequestDevice(g_adapter, &dd, dci);
        for(int g=0; !dr.done && g<1000000; ++g) wgpuInstanceProcessEvents(g_inst);
        dev = dr.device;
        if(dev && attempt==0) hasPass = 1;
    }
    *hasTSout = g_hasTS; *hasPassOut = hasPass;
    if(!dev) return 3;
    g_device = dev;
    g_queue  = wgpuDeviceGetQueue(g_device);
    g_period = g_hasTS ? wgpuQueueGetTimestampPeriod(g_queue) : 1.0f;
    return 0;
}

static WGPUBufferBindingType bbt(int kind){
    if(kind==0) return WGPUBufferBindingType_Uniform;
    if(kind==1) return WGPUBufferBindingType_ReadOnlyStorage;
    return WGPUBufferBindingType_Storage;
}

// np_pipeline builds an explicit BGL+PL from kinds[] (one per binding slot 0..nbind-1,
// 0=uniform 1=ro-storage 2=rw-storage), a shader module (from SPIR-V if fromSpirv,
// else WGSL), and a compute pipeline. Returns index or -1.
static int np_pipeline(int fromSpirv, const uint32_t* spv, int nWords, const char* wgsl, const int* kinds, int nbind){
    if(g_npipes >= NP_MAX_PIPE) return -1;
    WGPUBindGroupLayoutEntry* e = (WGPUBindGroupLayoutEntry*)calloc(nbind, sizeof(WGPUBindGroupLayoutEntry));
    for(int j=0;j<nbind;j++){ e[j].binding=(uint32_t)j; e[j].visibility=WGPUShaderStage_Compute; e[j].buffer.type=bbt(kinds[j]); }
    WGPUBindGroupLayoutDescriptor bld; memset(&bld,0,sizeof(bld)); bld.entryCount=(size_t)nbind; bld.entries=e;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(g_device,&bld);
    free(e);
    if(!bgl) return -1;
    WGPUPipelineLayoutDescriptor pld; memset(&pld,0,sizeof(pld)); pld.bindGroupLayoutCount=1; pld.bindGroupLayouts=&bgl;
    WGPUPipelineLayout pll = wgpuDeviceCreatePipelineLayout(g_device,&pld);
    if(!pll) return -1;

    WGPUShaderModule sh;
    if(fromSpirv){
        WGPUShaderModuleDescriptorSpirV sd; memset(&sd,0,sizeof(sd));
        sd.sourceSize=(uint32_t)nWords; sd.source=spv;
        sh = wgpuDeviceCreateShaderModuleSpirV(g_device,&sd);
    } else {
        WGPUShaderSourceWGSL w; memset(&w,0,sizeof(w)); w.chain.sType=WGPUSType_ShaderSourceWGSL; w.code=sv(wgsl);
        WGPUShaderModuleDescriptor smd; memset(&smd,0,sizeof(smd)); smd.nextInChain=&w.chain;
        sh = wgpuDeviceCreateShaderModule(g_device,&smd);
    }
    if(!sh) return -1;
    WGPUComputePipelineDescriptor cpd; memset(&cpd,0,sizeof(cpd));
    cpd.layout = pll; cpd.compute.module = sh; cpd.compute.entryPoint = sv("main");
    WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(g_device,&cpd);
    wgpuShaderModuleRelease(sh);
    if(!p) return -1;
    int i=g_npipes++; g_pipes[i]=p; g_bgls[i]=bgl; g_plls[i]=pll;
    return i;
}

// np_bench runs the K-dispatch dependent chain for pipeline idx. bindSlot/bindKind/
// bindStore describe the bind group: kind 0=uniform (uses the uniform buffer),
// else storage buffer #bindStore[j] (nStore allocated, bufElems f32 each). The
// rw storage buffer reused across all K dispatches creates 1 barrier/dispatch.
static int np_bench(int idx, int nStore, int bufElems,
                    const int* bindSlot, const int* bindKind, const int* bindStore, int nbind,
                    const void* uni, int uniLen, uint32_t groups, int K, int runs,
                    double* gpuMsOut){
    uint64_t bufBytes = (uint64_t)bufElems*4;
    WGPUBuffer* st = (WGPUBuffer*)calloc(nStore,sizeof(WGPUBuffer));
    for(int s=0;s<nStore;s++){
        WGPUBufferDescriptor bd; memset(&bd,0,sizeof(bd)); bd.size=bufBytes; bd.usage=WGPUBufferUsage_Storage|WGPUBufferUsage_CopyDst;
        st[s]=wgpuDeviceCreateBuffer(g_device,&bd);
    }
    WGPUBuffer ub=NULL;
    if(uniLen>0){
        WGPUBufferDescriptor bd; memset(&bd,0,sizeof(bd)); bd.size=(uint64_t)uniLen; bd.usage=WGPUBufferUsage_Uniform|WGPUBufferUsage_CopyDst;
        ub=wgpuDeviceCreateBuffer(g_device,&bd);
        wgpuQueueWriteBuffer(g_queue, ub, 0, uni, (size_t)uniLen);
    }
    WGPUBindGroupEntry* be = (WGPUBindGroupEntry*)calloc(nbind,sizeof(WGPUBindGroupEntry));
    for(int j=0;j<nbind;j++){
        be[j].binding=(uint32_t)bindSlot[j];
        if(bindKind[j]==0){ be[j].buffer=ub; be[j].size=(uint64_t)uniLen; }
        else { be[j].buffer=st[bindStore[j]]; be[j].size=bufBytes; }
    }
    WGPUBindGroupDescriptor bgd; memset(&bgd,0,sizeof(bgd)); bgd.layout=g_bgls[idx]; bgd.entryCount=(size_t)nbind; bgd.entries=be;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_device,&bgd);
    free(be);

    WGPUQuerySet qset=NULL; WGPUBuffer tsR=NULL,tsS=NULL;
    if(g_hasTS){
        WGPUQuerySetDescriptor qd; memset(&qd,0,sizeof(qd)); qd.type=WGPUQueryType_Timestamp; qd.count=2; qset=wgpuDeviceCreateQuerySet(g_device,&qd);
        WGPUBufferDescriptor rd; memset(&rd,0,sizeof(rd)); rd.size=16; rd.usage=WGPUBufferUsage_QueryResolve|WGPUBufferUsage_CopySrc; tsR=wgpuDeviceCreateBuffer(g_device,&rd);
        WGPUBufferDescriptor sd; memset(&sd,0,sizeof(sd)); sd.size=16; sd.usage=WGPUBufferUsage_MapRead|WGPUBufferUsage_CopyDst; tsS=wgpuDeviceCreateBuffer(g_device,&sd);
    }

    #define NP_ONCE(gpuOut) do { \
        WGPUCommandEncoder enc=wgpuDeviceCreateCommandEncoder(g_device,NULL); \
        WGPUPassTimestampWrites tw; memset(&tw,0,sizeof(tw)); tw.querySet=qset; tw.beginningOfPassWriteIndex=0; tw.endOfPassWriteIndex=1; \
        WGPUComputePassDescriptor pd; memset(&pd,0,sizeof(pd)); if(g_hasTS) pd.timestampWrites=&tw; \
        WGPUComputePassEncoder pass=wgpuCommandEncoderBeginComputePass(enc,&pd); \
        wgpuComputePassEncoderSetPipeline(pass,g_pipes[idx]); \
        for(int i=0;i<K;i++){ wgpuComputePassEncoderSetBindGroup(pass,0,bg,0,NULL); wgpuComputePassEncoderDispatchWorkgroups(pass,groups,1,1);} \
        wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass); \
        if(g_hasTS){ wgpuCommandEncoderResolveQuerySet(enc,qset,0,2,tsR,0); wgpuCommandEncoderCopyBufferToBuffer(enc,tsR,0,tsS,0,16);} \
        WGPUCommandBuffer cmd=wgpuCommandEncoderFinish(enc,NULL); wgpuQueueSubmit(g_queue,1,&cmd); wgpuCommandBufferRelease(cmd); wgpuCommandEncoderRelease(enc); \
        wgpuDevicePoll(g_device,1,NULL); gpuOut=0.0; \
        if(g_hasTS){ mapRes mr; mr.status=0; mr.done=0; WGPUBufferMapCallbackInfo ci; memset(&ci,0,sizeof(ci)); ci.mode=WGPUCallbackMode_AllowProcessEvents; ci.callback=mapCB; ci.userdata1=&mr; \
            wgpuBufferMapAsync(tsS,WGPUMapMode_Read,0,16,ci); wgpuDevicePoll(g_device,1,NULL); \
            const uint64_t* ts=(const uint64_t*)wgpuBufferGetMappedRange(tsS,0,16); if(ts&&ts[1]>=ts[0]) gpuOut=(double)(ts[1]-ts[0])*(double)g_period/1e6; wgpuBufferUnmap(tsS);} \
    } while(0)

    double g; NP_ONCE(g); // warmup
    double* gpu=(double*)calloc(runs,sizeof(double));
    for(int r=0;r<runs;r++){ double gg; NP_ONCE(gg); gpu[r]=gg; }
    for(int a=1;a<runs;a++){ for(int b=a;b>0&&gpu[b-1]>gpu[b];b--){double t=gpu[b-1];gpu[b-1]=gpu[b];gpu[b]=t;} }
    *gpuMsOut = gpu[runs/2];
    free(gpu);

    if(bg) wgpuBindGroupRelease(bg);
    for(int s=0;s<nStore;s++) if(st[s]) wgpuBufferRelease(st[s]);
    free(st);
    if(ub) wgpuBufferRelease(ub);
    if(qset) wgpuQuerySetRelease(qset);
    if(tsR) wgpuBufferRelease(tsR);
    if(tsS) wgpuBufferRelease(tsS);
    return 0;
}
*/
import "C"

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"
	"unsafe"
)

// fixture describes one glue kernel's binding layout for the chain harness.
// kinds: 0=uniform, 1=read-only storage, 2=read-write storage. store: storage
// buffer index for storage bindings (-1 for uniform); aliasing an ro input and
// the rw output to the same store index makes the chain a real RAW dependency.
type fixture struct {
	name   string
	kinds  []int    // by binding slot 0..n-1 (for the BGL)
	store  []int    // storage buffer index per slot (-1 = uniform)
	nStore int      // distinct storage buffers
	uni    []uint32 // uniform P contents (4 words)
	groups uint32
}

const bufElems = 4096

// The chain dependency comes from the rw storage buffer being reused across all
// K dispatches (write-after-write hazard => 1 barrier/dispatch, serialized) — no
// buffer is bound to two slots (illegal aliasing). The kernel still does its full
// real work each dispatch; the (a)/(b) delta isolates codegen since the barrier
// is constant across blobs.
var fixtures = map[string]fixture{
	// rmsnorm: src(0,ro) weight(1,ro) dst(2,rw) p(3,uni). dst reused => chain.
	"rmsnorm": {"rmsnorm", []int{1, 1, 2, 0}, []int{0, 1, 2, -1}, 3, []uint32{256, math.Float32bits(1e-5), 0, 0}, 1},
	// qknorm: vec(0,rw) weight(1,ro) p(2,uni). vec rw reused => chain.
	"qknorm": {"qknorm", []int{2, 1, 0}, []int{0, 1, -1}, 2, []uint32{1, 256, math.Float32bits(1e-5), 0}, 1},
	// residual: x(0,rw) y(1,ro) p(2,uni). x rw reused => chain.
	"residual": {"residual", []int{2, 1, 0}, []int{0, 1, -1}, 2, []uint32{256, 0, 0, 0}, 4},
	// swiglu: gate(0,ro) up(1,ro) dst(2,rw) p(3,uni). dst reused => chain.
	"swiglu": {"swiglu", []int{1, 1, 2, 0}, []int{0, 1, 2, -1}, 3, []uint32{256, 0, 0, 0}, 4},
}

func main() {
	backend := flag.String("backend", "vulkan", "backend (vulkan|metal|...)")
	fixName := flag.String("fixture", "", "fixture name: rmsnorm|qknorm|residual|swiglu")
	spvPath := flag.String("spv", "", "path to precompiled SPIR-V (passthrough); mutually exclusive with -wgsl")
	wgslPath := flag.String("wgsl", "", "path to WGSL (compiled by the runtime's own naga; sanity arm)")
	label := flag.String("label", "", "label for the CSV row (e.g. naga22, naga29, wgsl)")
	k := flag.Int("k", 400, "chain length")
	runs := flag.Int("runs", 50, "timed reps (median)")
	csv := flag.Bool("csv", false, "emit csv row")
	flag.Parse()

	fx, ok := fixtures[*fixName]
	if !ok {
		fmt.Fprintf(os.Stderr, "nagaprobe: unknown -fixture %q (have rmsnorm|qknorm|residual|swiglu)\n", *fixName)
		os.Exit(1)
	}

	var name [256]C.char
	var hasPass, hasTS C.int
	var ver C.uint32_t
	if rc := C.np_init(backendCode(*backend), &name[0], 256, &hasPass, &hasTS, &ver); rc != 0 {
		fmt.Fprintf(os.Stderr, "nagaprobe: np_init failed rc=%d\n", int(rc))
		os.Exit(1)
	}
	fmt.Printf("wgpu-native 0x%08x | adapter: %s | timestamps=%v | spirv-passthrough=%v\n",
		uint32(ver), C.GoString(&name[0]), hasTS != 0, hasPass != 0)
	if *spvPath != "" && hasPass == 0 {
		fmt.Fprintln(os.Stderr, "nagaprobe: adapter lacks SpirvShaderPassthrough; cannot run -spv")
		os.Exit(1)
	}

	// kinds (BGL order) for np_pipeline.
	kinds := fx.kinds
	var pidx C.int
	if *spvPath != "" {
		words, err := readSpirv(*spvPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "nagaprobe: %v\n", err)
			os.Exit(1)
		}
		pidx = C.np_pipeline(1, (*C.uint32_t)(unsafe.Pointer(&words[0])), C.int(len(words)), nil,
			(*C.int)(unsafe.Pointer(&toC(kinds)[0])), C.int(len(kinds)))
	} else if *wgslPath != "" {
		src, err := os.ReadFile(*wgslPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "nagaprobe: %v\n", err)
			os.Exit(1)
		}
		csrc := C.CString(string(src))
		defer C.free(unsafe.Pointer(csrc))
		pidx = C.np_pipeline(0, nil, 0, csrc, (*C.int)(unsafe.Pointer(&toC(kinds)[0])), C.int(len(kinds)))
	} else {
		fmt.Fprintln(os.Stderr, "nagaprobe: need -spv or -wgsl")
		os.Exit(1)
	}
	if pidx < 0 {
		fmt.Fprintln(os.Stderr, "nagaprobe: pipeline creation failed (compile/passthrough error)")
		os.Exit(1)
	}

	// bind arrays for np_bench.
	nbind := len(fx.kinds)
	slot := make([]int, nbind)
	for i := range slot {
		slot[i] = i
	}
	uniBytes := wordsToBytes(fx.uni)

	var gpuMs C.double
	C.np_bench(pidx, C.int(fx.nStore), C.int(bufElems),
		(*C.int)(unsafe.Pointer(&toC(slot)[0])),
		(*C.int)(unsafe.Pointer(&toC(fx.kinds)[0])),
		(*C.int)(unsafe.Pointer(&toC(fx.store)[0])), C.int(nbind),
		unsafe.Pointer(&uniBytes[0]), C.int(len(uniBytes)),
		C.uint32_t(fx.groups), C.int(*k), C.int(*runs), &gpuMs)

	perDisp := float64(gpuMs) * 1000.0 / float64(*k)
	lab := *label
	if lab == "" {
		lab = "?"
	}
	fmt.Printf("%-10s %-8s K=%d  gpu=%.4f ms  per-dispatch=%.4f us\n", fx.name, lab, *k, float64(gpuMs), perDisp)
	if *csv {
		fmt.Printf("csv,0x%08x,%s,%s,%d,%.4f,%.4f\n", uint32(ver), fx.name, lab, *k, float64(gpuMs), perDisp)
	}
}

func backendCode(s string) C.uint32_t {
	switch strings.ToLower(s) {
	case "vulkan", "vk":
		return C.WGPUBackendType_Vulkan
	case "metal":
		return C.WGPUBackendType_Metal
	default:
		return C.WGPUBackendType_Undefined
	}
}

// toC copies an int slice to a C.int slice (sizes may differ across platforms).
func toC(xs []int) []C.int {
	out := make([]C.int, len(xs))
	for i, v := range xs {
		out[i] = C.int(v)
	}
	return out
}

func wordsToBytes(w []uint32) []byte {
	b := make([]byte, len(w)*4)
	for i, v := range w {
		b[i*4] = byte(v)
		b[i*4+1] = byte(v >> 8)
		b[i*4+2] = byte(v >> 16)
		b[i*4+3] = byte(v >> 24)
	}
	return b
}

func readSpirv(path string) ([]uint32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 || len(b) == 0 {
		return nil, fmt.Errorf("%s: not a SPIR-V blob (%d bytes)", path, len(b))
	}
	w := make([]uint32, len(b)/4)
	for i := range w {
		w[i] = uint32(b[i*4]) | uint32(b[i*4+1])<<8 | uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24
	}
	return w, nil
}
