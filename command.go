//go:build cgo

package wgpu

/*
#include <stdlib.h>
#include <stdint.h>
#include "./lib/wgpu.h"

// cgoRecordSteps records a batch of compute dispatches into a pass with the loop
// running entirely in C, so the whole batch costs ONE Go->C crossing instead of
// ~3 per dispatch. Each step binds its pipeline + bind group at group 0 (no
// dynamic offsets) and dispatches (x,y,z); xyz is n*3 packed. Redundant
// SetPipeline calls (same as previous step) are skipped.
static void cgoRecordSteps(WGPUComputePassEncoder pass, size_t n,
                           WGPUComputePipeline const *pls,
                           WGPUBindGroup const *bgs,
                           uint32_t const *xyz) {
    WGPUComputePipeline cur = NULL;
    for (size_t i = 0; i < n; i++) {
        if (pls[i] != cur) { wgpuComputePassEncoderSetPipeline(pass, pls[i]); cur = pls[i]; }
        wgpuComputePassEncoderSetBindGroup(pass, 0, bgs[i], 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass, xyz[i*3], xyz[i*3+1], xyz[i*3+2]);
    }
}
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

// ComputeStep is one dispatch in a RecordSteps batch: bind Pipeline + BindGroup
// at group 0 (no dynamic offsets) and dispatch (X,Y,Z) workgroups. It models the
// common serial-decode-spine step (one bind group at group 0); use the per-call
// SetPipeline/SetBindGroup/DispatchWorkgroups methods directly if you need a
// non-zero group index or dynamic offsets.
type ComputeStep struct {
	Pipeline  *ComputePipeline
	BindGroup *BindGroup
	X, Y, Z   uint32
}

// RecordSteps records a batch of dispatches into the pass in a SINGLE cgo
// crossing — the per-step SetPipeline/SetBindGroup/DispatchWorkgroups loop runs
// in C. This is an additive, non-cogentcore-mirroring fast path for hot recorders
// (e.g. a ~hundreds-of-dispatch decode chain): equivalent to calling the three
// per-call methods for each step in order, but collapses ~3*len(steps) Go->C
// crossings into one. Consecutive steps that share a pipeline skip the redundant
// SetPipeline. A nil Pipeline or BindGroup in any step panics (programmer error).
func (p *ComputePassEncoder) RecordSteps(steps []ComputeStep) {
	n := len(steps)
	if n == 0 {
		return
	}
	pls := make([]C.WGPUComputePipeline, n)
	bgs := make([]C.WGPUBindGroup, n)
	xyz := make([]C.uint32_t, n*3)
	for i := range steps {
		s := &steps[i]
		if s.Pipeline == nil || s.BindGroup == nil {
			panic("wgpu: RecordSteps: step has nil Pipeline or BindGroup")
		}
		pls[i] = s.Pipeline.ref
		bgs[i] = s.BindGroup.ref
		xyz[i*3] = C.uint32_t(s.X)
		xyz[i*3+1] = C.uint32_t(s.Y)
		xyz[i*3+2] = C.uint32_t(s.Z)
	}
	C.cgoRecordSteps(p.ref, C.size_t(n), &pls[0], &bgs[0], &xyz[0])
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
