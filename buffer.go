//go:build cgo

package wgpu

/*
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "./lib/wgpu.h"

extern void cgoBufferMapCB(WGPUMapAsyncStatus status, WGPUStringView message, void *ud1, void *ud2);
*/
import "C"

import (
	"errors"
	"runtime/cgo"
	"unsafe"
)

// Buffer mirrors cogentcore/webgpu.Buffer.
type Buffer struct {
	ref C.WGPUBuffer
}

// BufferDescriptor mirrors cogentcore.
type BufferDescriptor struct {
	Label            string
	Usage            BufferUsage
	Size             uint64
	MappedAtCreation bool
}

// BufferInitDescriptor mirrors cogentcore: a buffer created with initial
// contents (uploaded via mappedAtCreation).
type BufferInitDescriptor struct {
	Label    string
	Contents []byte
	Usage    BufferUsage
}

// CreateBuffer creates an empty buffer.
func (d *Device) CreateBuffer(descriptor *BufferDescriptor) (*Buffer, error) {
	if descriptor == nil {
		return nil, errors.New("wgpu: CreateBuffer requires a descriptor")
	}
	clearLastError()
	var desc C.WGPUBufferDescriptor
	if descriptor.Label != "" {
		lbl := newStringView(descriptor.Label)
		defer freeStringView(lbl)
		desc.label = lbl
	}
	desc.usage = C.WGPUBufferUsage(descriptor.Usage)
	desc.size = C.uint64_t(descriptor.Size)
	desc.mappedAtCreation = cBool(descriptor.MappedAtCreation)
	ref := C.wgpuDeviceCreateBuffer(d.ref, &desc)
	if ref == nil {
		return nil, errors.New("wgpu: CreateBuffer failed: " + errOr(takeLastError(), "validation error"))
	}
	return &Buffer{ref: ref}, nil
}

// CreateBufferInit creates a buffer initialised with Contents. The size is
// rounded up to a multiple of 4 (COPY/map alignment). Empty contents yields a
// zero-size buffer error to match practical usage.
func (d *Device) CreateBufferInit(descriptor *BufferInitDescriptor) (*Buffer, error) {
	if descriptor == nil {
		return nil, errors.New("wgpu: CreateBufferInit requires a descriptor")
	}
	contentLen := len(descriptor.Contents)
	if contentLen == 0 {
		return nil, errors.New("wgpu: CreateBufferInit requires non-empty Contents")
	}
	// Round size up to 4 bytes (wgpu requires mapped/copy sizes aligned to 4).
	size := uint64(contentLen)
	if rem := size % 4; rem != 0 {
		size += 4 - rem
	}
	buf, err := d.CreateBuffer(&BufferDescriptor{
		Label:            descriptor.Label,
		Usage:            descriptor.Usage,
		Size:             size,
		MappedAtCreation: true,
	})
	if err != nil {
		return nil, err
	}
	// Copy contents into the mapped range, then unmap.
	dst := C.wgpuBufferGetMappedRange(buf.ref, 0, C.size_t(size))
	if dst == nil {
		buf.Release()
		return nil, errors.New("wgpu: CreateBufferInit: GetMappedRange returned nil")
	}
	C.memcpy(dst, unsafe.Pointer(&descriptor.Contents[0]), C.size_t(contentLen))
	C.wgpuBufferUnmap(buf.ref)
	return buf, nil
}

// GetSize returns the buffer's size in bytes.
func (p *Buffer) GetSize() uint64 { return uint64(C.wgpuBufferGetSize(p.ref)) }

// GetUsage returns the buffer's usage flags.
func (p *Buffer) GetUsage() BufferUsage { return BufferUsage(C.wgpuBufferGetUsage(p.ref)) }

// GetMappedRange returns a Go slice aliasing the buffer's mapped CPU range.
// Valid only while the buffer is mapped.
func (p *Buffer) GetMappedRange(offset, size uint) []byte {
	ptr := C.wgpuBufferGetMappedRange(p.ref, C.size_t(offset), C.size_t(size))
	if ptr == nil {
		return nil
	}
	return unsafe.Slice((*byte)(ptr), size)
}

// MapAsync requests an async CPU map. callback fires during a subsequent
// Device.Poll. Matches cogentcore's signature.
func (p *Buffer) MapAsync(mode MapMode, offset uint64, size uint64, callback BufferMapCallback) error {
	// Stash the handle in a malloc'd cell so its lifetime is independent of the
	// Go stack (the callback fires later, inside Poll).
	h := cgo.NewHandle(callback)
	cell := C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	*(*C.uintptr_t)(cell) = C.uintptr_t(h)

	var ci C.WGPUBufferMapCallbackInfo
	ci.mode = C.WGPUCallbackMode_AllowProcessEvents
	ci.callback = C.WGPUBufferMapCallback(C.cgoBufferMapCB)
	ci.userdata1 = cell

	C.wgpuBufferMapAsync(p.ref, C.WGPUMapMode(mode), C.size_t(offset), C.size_t(size), ci)
	return nil
}

// Unmap unmaps a previously mapped buffer.
func (p *Buffer) Unmap() error {
	C.wgpuBufferUnmap(p.ref)
	return nil
}

// Destroy frees the buffer's GPU memory.
func (p *Buffer) Destroy() { C.wgpuBufferDestroy(p.ref) }

// Release drops the buffer reference.
func (p *Buffer) Release() { C.wgpuBufferRelease(p.ref) }
