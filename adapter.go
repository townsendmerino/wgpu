//go:build cgo

package wgpu

/*
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "./lib/wgpu.h"

extern void cgoUncapturedErrorCB(WGPUDevice const *device, WGPUErrorType type, WGPUStringView message, void *ud1, void *ud2);

// Drive wgpuAdapterRequestDevice to completion synchronously (see instance.go).
typedef struct { uint32_t status; WGPUDevice device; int done; } cgoDeviceResult;

static void cgoDeviceCB(WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView m, void *u1, void *u2) {
    (void)m; (void)u2;
    cgoDeviceResult *r = (cgoDeviceResult *)u1;
    r->status = (uint32_t)s;
    r->device = d;
    r->done = 1;
}

static WGPUDevice cgoRequestDeviceSync(WGPUInstance inst, WGPUAdapter ad, WGPUDeviceDescriptor const *desc, uint32_t *status) {
    cgoDeviceResult r;
    r.status = 0; r.device = NULL; r.done = 0;
    WGPURequestDeviceCallbackInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.mode = WGPUCallbackMode_AllowProcessEvents;
    ci.callback = cgoDeviceCB;
    ci.userdata1 = &r;
    wgpuAdapterRequestDevice(ad, desc, ci);
    int guard = 0;
    while (!r.done && guard++ < 1000000) {
        wgpuInstanceProcessEvents(inst);
    }
    *status = r.status;
    return r.device;
}
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Adapter mirrors cogentcore/webgpu.Adapter.
type Adapter struct {
	ref         C.WGPUAdapter
	instanceRef C.WGPUInstance // needed to pump events while requesting a device
}

// AdapterInfo mirrors cogentcore's AdapterInfo, plus v29 subgroup sizes.
type AdapterInfo struct {
	VendorId          uint32
	VendorName        string
	Architecture      string
	DeviceId          uint32
	Name              string
	DriverDescription string
	AdapterType       AdapterType
	BackendType       BackendType
	SubgroupMinSize   uint32 // v29
	SubgroupMaxSize   uint32 // v29
}

// GetInfo returns identifying information about the adapter.
func (p *Adapter) GetInfo() AdapterInfo {
	var info C.WGPUAdapterInfo
	C.wgpuAdapterGetInfo(p.ref, &info)
	out := AdapterInfo{
		VendorId:          uint32(info.vendorID),
		VendorName:        goStringView(info.vendor),
		Architecture:      goStringView(info.architecture),
		DeviceId:          uint32(info.deviceID),
		Name:              goStringView(info.device),
		DriverDescription: goStringView(info.description),
		AdapterType:       AdapterType(info.adapterType),
		BackendType:       BackendType(info.backendType),
		SubgroupMinSize:   uint32(info.subgroupMinSize),
		SubgroupMaxSize:   uint32(info.subgroupMaxSize),
	}
	C.wgpuAdapterInfoFreeMembers(info)
	return out
}

// GetLimits returns the adapter's supported limits.
func (p *Adapter) GetLimits() SupportedLimits {
	var l C.WGPULimits
	C.wgpuAdapterGetLimits(p.ref, &l)
	return SupportedLimits{Limits: limitsFromC(l)}
}

// EnumerateFeatures returns the features the adapter supports (standard +
// native; native features share the WGPUFeatureName value space).
func (p *Adapter) EnumerateFeatures() []FeatureName {
	var sf C.WGPUSupportedFeatures
	C.wgpuAdapterGetFeatures(p.ref, &sf)
	n := int(sf.featureCount)
	if n == 0 {
		C.wgpuSupportedFeaturesFreeMembers(sf)
		return nil
	}
	src := unsafe.Slice((*C.WGPUFeatureName)(unsafe.Pointer(sf.features)), n)
	out := make([]FeatureName, n)
	for i := 0; i < n; i++ {
		out[i] = FeatureName(src[i])
	}
	C.wgpuSupportedFeaturesFreeMembers(sf)
	return out
}

// HasFeature reports whether the adapter supports a feature.
func (p *Adapter) HasFeature(feature FeatureName) bool {
	return goBool(C.wgpuAdapterHasFeature(p.ref, C.WGPUFeatureName(feature)))
}

// DeviceLostCallback is retained for source compatibility; it is not wired in
// the compute-only binding.
type DeviceLostCallback func(reason int, message string)

// DeviceDescriptor mirrors cogentcore's subset.
type DeviceDescriptor struct {
	Label              string
	RequiredFeatures   []FeatureName
	RequiredLimits     *RequiredLimits
	DeviceLostCallback DeviceLostCallback // accepted but not wired
	TracePath          string             // accepted but not wired
}

// RequestDevice synchronously requests a device. An uncaptured-error callback is
// installed so validation failures surface as real messages (and to stderr).
func (p *Adapter) RequestDevice(descriptor *DeviceDescriptor) (*Device, error) {
	clearLastError()
	var desc C.WGPUDeviceDescriptor

	// Uncaptured error callback (always installed).
	desc.uncapturedErrorCallbackInfo.callback = C.WGPUUncapturedErrorCallback(C.cgoUncapturedErrorCB)

	if descriptor != nil {
		if descriptor.Label != "" {
			lbl := newStringView(descriptor.Label)
			defer freeStringView(lbl)
			desc.label = lbl
		}
		if n := len(descriptor.RequiredFeatures); n > 0 {
			feats := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.WGPUFeatureName(0))))
			defer C.free(feats)
			fs := unsafe.Slice((*C.WGPUFeatureName)(feats), n)
			for i, f := range descriptor.RequiredFeatures {
				fs[i] = C.WGPUFeatureName(f)
			}
			desc.requiredFeatures = (*C.WGPUFeatureName)(feats)
			desc.requiredFeatureCount = C.size_t(n)
		}
		if descriptor.RequiredLimits != nil {
			cl := (*C.WGPULimits)(C.calloc(1, C.size_t(unsafe.Sizeof(C.WGPULimits{}))))
			defer C.free(unsafe.Pointer(cl))
			fillCLimits(cl, descriptor.RequiredLimits.Limits)
			desc.requiredLimits = cl
		}
	}

	var status C.uint32_t
	ref := C.cgoRequestDeviceSync(p.instanceRef, p.ref, &desc, &status)
	if ref == nil || uint32(status) != uint32(C.WGPURequestDeviceStatus_Success) {
		return nil, errors.New("wgpu: RequestDevice failed: " + errOr(takeLastError(), "unknown error"))
	}
	return &Device{ref: ref}, nil
}

func (p *Adapter) Release() { C.wgpuAdapterRelease(p.ref) }
