//go:build cgo

package wgpu

/*
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "./lib/wgpu.h"

// Drive wgpuInstanceRequestAdapter (async in v29) to completion synchronously.
// wgpu-native resolves the adapter request inside processEvents; the callback
// stores the result in a stack struct so no Go memory crosses the boundary.
typedef struct { uint32_t status; WGPUAdapter adapter; int done; } cgoAdapterResult;

static void cgoAdapterCB(WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView m, void *u1, void *u2) {
    (void)m; (void)u2;
    cgoAdapterResult *r = (cgoAdapterResult *)u1;
    r->status = (uint32_t)s;
    r->adapter = a;
    r->done = 1;
}

static WGPUAdapter cgoRequestAdapterSync(WGPUInstance inst, WGPURequestAdapterOptions const *opts, uint32_t *status) {
    cgoAdapterResult r;
    r.status = 0; r.adapter = NULL; r.done = 0;
    WGPURequestAdapterCallbackInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.mode = WGPUCallbackMode_AllowProcessEvents;
    ci.callback = cgoAdapterCB;
    ci.userdata1 = &r;
    wgpuInstanceRequestAdapter(inst, opts, ci);
    int guard = 0;
    while (!r.done && guard++ < 1000000) {
        wgpuInstanceProcessEvents(inst);
    }
    *status = r.status;
    return r.adapter;
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Instance is the entry point. Mirrors cogentcore/webgpu.Instance.
type Instance struct {
	ref C.WGPUInstance
}

// RequestAdapterOptions mirrors cogentcore's subset.
type RequestAdapterOptions struct {
	PowerPreference      PowerPreference
	ForceFallbackAdapter bool
	BackendType          BackendType
	// CompatibleSurface is unused in the compute-only binding (kept for source
	// compatibility); always nil.
	CompatibleSurface *struct{}
}

// CreateInstance creates a wgpu instance. descriptor may be nil (the common
// compute case), which uses wgpu-native defaults (all backends).
func CreateInstance(descriptor *InstanceDescriptor) *Instance {
	var desc *C.WGPUInstanceDescriptor
	// InstanceExtras (backends/flags) is only built when a descriptor is given.
	if descriptor != nil {
		extras := (*C.WGPUInstanceExtras)(C.calloc(1, C.size_t(unsafe.Sizeof(C.WGPUInstanceExtras{}))))
		defer C.free(unsafe.Pointer(extras))
		extras.chain.sType = C.WGPUSType_InstanceExtras
		extras.backends = C.WGPUInstanceBackend(descriptor.Backends)
		var d C.WGPUInstanceDescriptor
		d.nextInChain = &extras.chain
		desc = &d
	}
	ref := C.wgpuCreateInstance(desc)
	if ref == nil {
		panic("wgpu: failed to create Instance")
	}
	return &Instance{ref: ref}
}

// InstanceDescriptor is an optional descriptor for CreateInstance. For
// compute-only use, pass nil to CreateInstance.
type InstanceDescriptor struct {
	// Backends is a bitset of WGPUInstanceBackend flags (0 = all).
	Backends uint64
}

// RequestAdapter synchronously requests an adapter. v29's async future is driven
// to completion internally, matching cogentcore's blocking signature.
func (p *Instance) RequestAdapter(options *RequestAdapterOptions) (*Adapter, error) {
	var opts *C.WGPURequestAdapterOptions
	if options != nil {
		o := C.WGPURequestAdapterOptions{}
		o.powerPreference = C.WGPUPowerPreference(options.PowerPreference)
		o.forceFallbackAdapter = cBool(options.ForceFallbackAdapter)
		o.backendType = C.WGPUBackendType(options.BackendType)
		opts = &o
	}
	var status C.uint32_t
	ref := C.cgoRequestAdapterSync(p.ref, opts, &status)
	if ref == nil || uint32(status) != uint32(C.WGPURequestAdapterStatus_Success) {
		return nil, errors.New("wgpu: RequestAdapter failed (no suitable adapter)")
	}
	return &Adapter{ref: ref, instanceRef: p.ref}, nil
}

func (p *Instance) Release() { C.wgpuInstanceRelease(p.ref) }
