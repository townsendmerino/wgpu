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

// BindGroupEntry mirrors cogentcore. Only buffer bindings are used by the
// compute-only consumer; Sampler/TextureView are accepted for source
// compatibility but unused.
type BindGroupEntry struct {
	Binding     uint32
	Buffer      *Buffer
	Offset      uint64
	Size        uint64
	Sampler     *struct{} // unused (source compat)
	TextureView *struct{} // unused (source compat)
}

// BindGroupDescriptor mirrors cogentcore.
type BindGroupDescriptor struct {
	Label   string
	Layout  *BindGroupLayout
	Entries []BindGroupEntry
}

// CreateBindGroup creates a bind group.
func (d *Device) CreateBindGroup(descriptor *BindGroupDescriptor) (*BindGroup, error) {
	if descriptor == nil || descriptor.Layout == nil {
		return nil, errors.New("wgpu: CreateBindGroup requires a Layout")
	}
	clearLastError()
	var desc C.WGPUBindGroupDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}
	desc.layout = descriptor.Layout.ref

	if n := len(descriptor.Entries); n > 0 {
		arr := (*C.WGPUBindGroupEntry)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof(C.WGPUBindGroupEntry{}))))
		defer C.free(unsafe.Pointer(arr))
		es := unsafe.Slice(arr, n)
		for i, e := range descriptor.Entries {
			es[i].binding = C.uint32_t(e.Binding)
			if e.Buffer != nil {
				es[i].buffer = e.Buffer.ref
			}
			es[i].offset = C.uint64_t(e.Offset)
			es[i].size = C.uint64_t(e.Size)
		}
		desc.entryCount = C.size_t(n)
		desc.entries = arr
	}

	ref := C.wgpuDeviceCreateBindGroup(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateBindGroup failed: " + errOr(takeLastError(), "validation error"))
	}
	return &BindGroup{ref: ref}, nil
}

// ---- explicit bind group layout (spec-listed; unused by goinfer, which uses
// ComputePipeline.GetBindGroupLayout). Supports compute buffer bindings. ----

// BufferBindingType mirrors WGPUBufferBindingType.
type BufferBindingType uint32

const (
	BufferBindingTypeUniform         BufferBindingType = C.WGPUBufferBindingType_Uniform
	BufferBindingTypeStorage         BufferBindingType = C.WGPUBufferBindingType_Storage
	BufferBindingTypeReadOnlyStorage BufferBindingType = C.WGPUBufferBindingType_ReadOnlyStorage
)

// BufferBindingLayout mirrors cogentcore's subset.
type BufferBindingLayout struct {
	Type             BufferBindingType
	HasDynamicOffset bool
	MinBindingSize   uint64
}

// BindGroupLayoutEntry mirrors cogentcore's subset (buffer bindings, compute
// visibility).
type BindGroupLayoutEntry struct {
	Binding    uint32
	Visibility ShaderStage
	Buffer     BufferBindingLayout
}

// ShaderStage is a bitset of WGPUShaderStage flags.
type ShaderStage uint64

const (
	ShaderStageNone    ShaderStage = 0x0000000000000000
	ShaderStageCompute ShaderStage = 0x0000000000000004
)

// BindGroupLayoutDescriptor mirrors cogentcore's subset.
type BindGroupLayoutDescriptor struct {
	Label   string
	Entries []BindGroupLayoutEntry
}

// CreateBindGroupLayout creates an explicit bind group layout (buffer bindings).
func (d *Device) CreateBindGroupLayout(descriptor *BindGroupLayoutDescriptor) (*BindGroupLayout, error) {
	if descriptor == nil {
		return nil, errors.New("wgpu: CreateBindGroupLayout requires a descriptor")
	}
	clearLastError()
	var desc C.WGPUBindGroupLayoutDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}
	if n := len(descriptor.Entries); n > 0 {
		arr := (*C.WGPUBindGroupLayoutEntry)(C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof(C.WGPUBindGroupLayoutEntry{}))))
		defer C.free(unsafe.Pointer(arr))
		es := unsafe.Slice(arr, n)
		for i, e := range descriptor.Entries {
			es[i].binding = C.uint32_t(e.Binding)
			es[i].visibility = C.WGPUShaderStage(e.Visibility)
			es[i].buffer._type = C.WGPUBufferBindingType(e.Buffer.Type)
			es[i].buffer.hasDynamicOffset = cBool(e.Buffer.HasDynamicOffset)
			es[i].buffer.minBindingSize = C.uint64_t(e.Buffer.MinBindingSize)
		}
		desc.entryCount = C.size_t(n)
		desc.entries = arr
	}
	ref := C.wgpuDeviceCreateBindGroupLayout(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateBindGroupLayout failed: " + errOr(takeLastError(), "validation error"))
	}
	return &BindGroupLayout{ref: ref}, nil
}
