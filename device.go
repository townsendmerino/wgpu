//go:build cgo

package wgpu

/*
#include <stdlib.h>
#include <stdint.h>
#include "./lib/wgpu.h"
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Device mirrors cogentcore/webgpu.Device.
type Device struct {
	ref C.WGPUDevice
}

// ---- shader modules --------------------------------------------------------

// ShaderModuleWGSLDescriptor carries WGSL source. Field name matches cogentcore.
type ShaderModuleWGSLDescriptor struct {
	Code string
}

// ShaderModuleDescriptor mirrors cogentcore's subset (WGSL only).
type ShaderModuleDescriptor struct {
	Label          string
	WGSLDescriptor *ShaderModuleWGSLDescriptor
}

// CreateShaderModule compiles a WGSL shader module.
func (d *Device) CreateShaderModule(descriptor *ShaderModuleDescriptor) (*ShaderModule, error) {
	if descriptor == nil || descriptor.WGSLDescriptor == nil {
		return nil, errors.New("wgpu: CreateShaderModule requires a WGSLDescriptor")
	}
	clearLastError()

	var desc C.WGPUShaderModuleDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}

	// Chain the WGSL source on the C heap so desc holds no Go pointers.
	wgsl := (*C.WGPUShaderSourceWGSL)(C.calloc(1, C.size_t(unsafe.Sizeof(C.WGPUShaderSourceWGSL{}))))
	defer C.free(unsafe.Pointer(wgsl))
	wgsl.chain.sType = C.WGPUSType_ShaderSourceWGSL
	code := newStringView(descriptor.WGSLDescriptor.Code)
	defer freeStringView(code)
	wgsl.code = code
	desc.nextInChain = &wgsl.chain

	ref := C.wgpuDeviceCreateShaderModule(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateShaderModule failed: " + errOr(takeLastError(), "compile error"))
	}
	return &ShaderModule{ref: ref}, nil
}

// ---- compute pipelines -----------------------------------------------------

// ConstantEntry sets a WGSL pipeline-overridable constant (v29 extra).
type ConstantEntry struct {
	Key   string
	Value float64
}

// ProgrammableStageDescriptor mirrors cogentcore, plus optional override
// Constants (v29 extra) for tuning without editing shader source.
type ProgrammableStageDescriptor struct {
	Module     *ShaderModule
	EntryPoint string
	Constants  []ConstantEntry
}

// ComputePipelineDescriptor mirrors cogentcore. Layout nil ⇒ auto layout.
type ComputePipelineDescriptor struct {
	Label   string
	Layout  *PipelineLayout
	Compute ProgrammableStageDescriptor
}

// CreateComputePipeline creates a compute pipeline.
func (d *Device) CreateComputePipeline(descriptor *ComputePipelineDescriptor) (*ComputePipeline, error) {
	if descriptor == nil || descriptor.Compute.Module == nil {
		return nil, errors.New("wgpu: CreateComputePipeline requires a compute Module")
	}
	clearLastError()

	var desc C.WGPUComputePipelineDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}
	if descriptor.Layout != nil {
		desc.layout = descriptor.Layout.ref
	}
	desc.compute.module = descriptor.Compute.Module.ref
	if descriptor.Compute.EntryPoint != "" {
		ep := newStringView(descriptor.Compute.EntryPoint)
		defer freeStringView(ep)
		desc.compute.entryPoint = ep
	}
	if n := len(descriptor.Compute.Constants); n > 0 {
		arr := (*C.WGPUConstantEntry)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof(C.WGPUConstantEntry{}))))
		defer C.free(unsafe.Pointer(arr))
		entries := unsafe.Slice(arr, n)
		for i, ce := range descriptor.Compute.Constants {
			key := newStringView(ce.Key)
			defer freeStringView(key)
			entries[i].key = key
			entries[i].value = C.double(ce.Value)
		}
		desc.compute.constantCount = C.size_t(n)
		desc.compute.constants = arr
	}

	ref := C.wgpuDeviceCreateComputePipeline(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateComputePipeline failed: " + errOr(takeLastError(), "validation error"))
	}
	return &ComputePipeline{ref: ref}, nil
}

// ---- command encoders ------------------------------------------------------

// CommandEncoderDescriptor is optional (label only).
type CommandEncoderDescriptor struct {
	Label string
}

// CreateCommandEncoder creates a command encoder. descriptor may be nil.
func (d *Device) CreateCommandEncoder(descriptor *CommandEncoderDescriptor) (*CommandEncoder, error) {
	var desc *C.WGPUCommandEncoderDescriptor
	if descriptor != nil && descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc = &C.WGPUCommandEncoderDescriptor{label: lbl}
	}
	ref := C.wgpuDeviceCreateCommandEncoder(d.ref, desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateCommandEncoder failed")
	}
	return &CommandEncoder{ref: ref}, nil
}

// ---- query sets (timestamps) -----------------------------------------------

// QuerySetDescriptor describes a query set (v29 extra, for timestamps).
type QuerySetDescriptor struct {
	Label string
	Type  QueryType
	Count uint32
}

// CreateQuerySet creates a query set (e.g. for timestamp queries).
func (d *Device) CreateQuerySet(descriptor *QuerySetDescriptor) (*QuerySet, error) {
	if descriptor == nil {
		return nil, errors.New("wgpu: CreateQuerySet requires a descriptor")
	}
	clearLastError()
	var desc C.WGPUQuerySetDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}
	desc._type = C.WGPUQueryType(descriptor.Type)
	desc.count = C.uint32_t(descriptor.Count)
	ref := C.wgpuDeviceCreateQuerySet(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateQuerySet failed: " + errOr(takeLastError(), "validation error"))
	}
	return &QuerySet{ref: ref}, nil
}

// ---- misc ------------------------------------------------------------------

// GetQueue returns the device's default queue.
func (d *Device) GetQueue() *Queue {
	return &Queue{ref: C.wgpuDeviceGetQueue(d.ref), deviceRef: d.ref}
}

// HasFeature reports whether the device has a feature enabled.
func (d *Device) HasFeature(feature FeatureName) bool {
	return goBool(C.wgpuDeviceHasFeature(d.ref, C.WGPUFeatureName(feature)))
}

// GetLimits returns the device's limits.
func (d *Device) GetLimits() SupportedLimits {
	var l C.WGPULimits
	C.wgpuDeviceGetLimits(d.ref, &l)
	return SupportedLimits{Limits: limitsFromC(l)}
}

// WrappedSubmissionIndex identifies a specific queue submission to wait on in
// Poll. Matches cogentcore's shape; the Queue field is retained for source
// compatibility.
type WrappedSubmissionIndex struct {
	Queue           *Queue
	SubmissionIndex SubmissionIndex
}

// Poll processes pending work. With wait=true it blocks until the queue is
// empty (or, if wrappedSubmissionIndex is non-nil, until that submission
// completes), firing any pending buffer-map callbacks. Returns whether the
// queue is now empty.
func (d *Device) Poll(wait bool, wrappedSubmissionIndex *WrappedSubmissionIndex) bool {
	if wrappedSubmissionIndex != nil {
		idx := C.WGPUSubmissionIndex(wrappedSubmissionIndex.SubmissionIndex)
		return goBool(C.wgpuDevicePoll(d.ref, cBool(wait), &idx))
	}
	return goBool(C.wgpuDevicePoll(d.ref, cBool(wait), nil))
}

func (d *Device) Release() { C.wgpuDeviceRelease(d.ref) }
