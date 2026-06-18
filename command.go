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

// ComputePipeline mirrors cogentcore/webgpu.ComputePipeline.
type ComputePipeline struct {
	ref C.WGPUComputePipeline
}

// GetBindGroupLayout returns the auto-generated bind group layout for a group.
func (p *ComputePipeline) GetBindGroupLayout(groupIndex uint32) *BindGroupLayout {
	return &BindGroupLayout{ref: C.wgpuComputePipelineGetBindGroupLayout(p.ref, C.uint32_t(groupIndex))}
}

func (p *ComputePipeline) Release() { C.wgpuComputePipelineRelease(p.ref) }

// ---- command encoder -------------------------------------------------------

// CommandEncoder mirrors cogentcore/webgpu.CommandEncoder.
type CommandEncoder struct {
	ref C.WGPUCommandEncoder
}

// ComputePassTimestampWrites attaches timestamp queries to a compute pass
// (v29 extra). Use QuerySetIndexUndefined to skip an endpoint.
type ComputePassTimestampWrites struct {
	QuerySet                  *QuerySet
	BeginningOfPassWriteIndex uint32
	EndOfPassWriteIndex       uint32
}

// QuerySetIndexUndefined marks a timestamp write endpoint as unused.
const QuerySetIndexUndefined uint32 = 0xffffffff

// ComputePassDescriptor mirrors cogentcore, plus optional TimestampWrites.
type ComputePassDescriptor struct {
	Label           string
	TimestampWrites *ComputePassTimestampWrites
}

// BeginComputePass begins a compute pass. descriptor may be nil.
func (p *CommandEncoder) BeginComputePass(descriptor *ComputePassDescriptor) *ComputePassEncoder {
	var desc *C.WGPUComputePassDescriptor
	if descriptor != nil {
		var d C.WGPUComputePassDescriptor
		if descriptor.Label != "" {
			lbl := newStringView(descriptor.Label)
			defer freeStringView(lbl)
			d.label = lbl
		}
		if tw := descriptor.TimestampWrites; tw != nil {
			ctw := (*C.WGPUPassTimestampWrites)(C.calloc(1, C.size_t(unsafe.Sizeof(C.WGPUPassTimestampWrites{}))))
			defer C.free(unsafe.Pointer(ctw))
			if tw.QuerySet != nil {
				ctw.querySet = tw.QuerySet.ref
			}
			ctw.beginningOfPassWriteIndex = C.uint32_t(tw.BeginningOfPassWriteIndex)
			ctw.endOfPassWriteIndex = C.uint32_t(tw.EndOfPassWriteIndex)
			d.timestampWrites = ctw
		}
		desc = &d
	}
	ref := C.wgpuCommandEncoderBeginComputePass(p.ref, desc)
	if ref == nil {
		panic("wgpu: failed to begin compute pass")
	}
	return &ComputePassEncoder{ref: ref}
}

// CopyBufferToBuffer records a buffer-to-buffer copy.
func (p *CommandEncoder) CopyBufferToBuffer(source *Buffer, sourceOffset uint64, destination *Buffer, destinationOffset uint64, size uint64) error {
	C.wgpuCommandEncoderCopyBufferToBuffer(p.ref, source.ref, C.uint64_t(sourceOffset),
		destination.ref, C.uint64_t(destinationOffset), C.uint64_t(size))
	return nil
}

// ResolveQuerySet copies query results into a buffer (v29 extra; for reading
// back timestamps). destination must include BufferUsageQueryResolve.
func (p *CommandEncoder) ResolveQuerySet(querySet *QuerySet, firstQuery, queryCount uint32, destination *Buffer, destinationOffset uint64) {
	C.wgpuCommandEncoderResolveQuerySet(p.ref, querySet.ref, C.uint32_t(firstQuery),
		C.uint32_t(queryCount), destination.ref, C.uint64_t(destinationOffset))
}

// CommandBufferDescriptor is optional (label only).
type CommandBufferDescriptor struct {
	Label string
}

// Finish finalizes the encoder into a command buffer. descriptor may be nil.
func (p *CommandEncoder) Finish(descriptor *CommandBufferDescriptor) (*CommandBuffer, error) {
	var desc *C.WGPUCommandBufferDescriptor
	if descriptor != nil && descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc = &C.WGPUCommandBufferDescriptor{label: lbl}
	}
	ref := C.wgpuCommandEncoderFinish(p.ref, desc)
	if ref == nil {
		return nil, errors.New("wgpu: CommandEncoder.Finish failed")
	}
	return &CommandBuffer{ref: ref}, nil
}

func (p *CommandEncoder) Release() { C.wgpuCommandEncoderRelease(p.ref) }

// ---- compute pass encoder --------------------------------------------------

// ComputePassEncoder mirrors cogentcore/webgpu.ComputePassEncoder.
type ComputePassEncoder struct {
	ref C.WGPUComputePassEncoder
}

// SetPipeline binds the compute pipeline.
func (p *ComputePassEncoder) SetPipeline(pipeline *ComputePipeline) {
	C.wgpuComputePassEncoderSetPipeline(p.ref, pipeline.ref)
}

// SetBindGroup binds a bind group. dynamicOffsets may be nil.
func (p *ComputePassEncoder) SetBindGroup(groupIndex uint32, group *BindGroup, dynamicOffsets []uint32) {
	if n := len(dynamicOffsets); n > 0 {
		C.wgpuComputePassEncoderSetBindGroup(p.ref, C.uint32_t(groupIndex), group.ref,
			C.size_t(n), (*C.uint32_t)(unsafe.Pointer(&dynamicOffsets[0])))
		return
	}
	C.wgpuComputePassEncoderSetBindGroup(p.ref, C.uint32_t(groupIndex), group.ref, 0, nil)
}

// DispatchWorkgroups dispatches a compute grid.
func (p *ComputePassEncoder) DispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ uint32) {
	C.wgpuComputePassEncoderDispatchWorkgroups(p.ref, C.uint32_t(workgroupCountX),
		C.uint32_t(workgroupCountY), C.uint32_t(workgroupCountZ))
}

// SetImmediates writes push-constant-equivalent "immediate" data (v29 extra;
// requires NativeFeatureImmediates and a non-zero maxImmediateSize limit).
func (p *ComputePassEncoder) SetImmediates(offset uint32, data []byte) {
	if len(data) == 0 {
		return
	}
	C.wgpuComputePassEncoderSetImmediates(p.ref, C.uint32_t(offset),
		C.uint32_t(len(data)), unsafe.Pointer(&data[0]))
}

// End ends the compute pass.
func (p *ComputePassEncoder) End() error {
	C.wgpuComputePassEncoderEnd(p.ref)
	return nil
}

func (p *ComputePassEncoder) Release() { C.wgpuComputePassEncoderRelease(p.ref) }
