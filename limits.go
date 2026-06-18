//go:build cgo

package wgpu

/*
#include "./lib/wgpu.h"
*/
import "C"

const (
	// LimitU32Undefined / LimitU64Undefined are the "no requirement / use
	// default" sentinels for limit fields (UINT32_MAX / UINT64_MAX in v29).
	LimitU32Undefined uint32 = 0xffffffff
	LimitU64Undefined uint64 = 0xffffffffffffffff
)

// Limits mirrors cogentcore/webgpu's Limits (field names preserved for drop-in
// source compatibility) with v29 additions. MaxInterStageShaderComponents is
// retained for source compatibility but no longer exists in v29 and is ignored
// when building C limits. MaxPushConstantSize maps to v29's maxImmediateSize.
type Limits struct {
	MaxTextureDimension1D                     uint32
	MaxTextureDimension2D                     uint32
	MaxTextureDimension3D                     uint32
	MaxTextureArrayLayers                     uint32
	MaxBindGroups                             uint32
	MaxBindGroupsPlusVertexBuffers            uint32 // v29
	MaxBindingsPerBindGroup                   uint32
	MaxDynamicUniformBuffersPerPipelineLayout uint32
	MaxDynamicStorageBuffersPerPipelineLayout uint32
	MaxSampledTexturesPerShaderStage          uint32
	MaxSamplersPerShaderStage                 uint32
	MaxStorageBuffersPerShaderStage           uint32
	MaxStorageTexturesPerShaderStage          uint32
	MaxUniformBuffersPerShaderStage           uint32
	MaxUniformBufferBindingSize               uint64
	MaxStorageBufferBindingSize               uint64
	MinUniformBufferOffsetAlignment           uint32
	MinStorageBufferOffsetAlignment           uint32
	MaxVertexBuffers                          uint32
	MaxBufferSize                             uint64
	MaxVertexAttributes                       uint32
	MaxVertexBufferArrayStride                uint32
	MaxInterStageShaderComponents             uint32 // deprecated/removed in v29 (ignored)
	MaxInterStageShaderVariables              uint32
	MaxColorAttachments                       uint32
	MaxColorAttachmentBytesPerSample          uint32
	MaxComputeWorkgroupStorageSize            uint32
	MaxComputeInvocationsPerWorkgroup         uint32
	MaxComputeWorkgroupSizeX                  uint32
	MaxComputeWorkgroupSizeY                  uint32
	MaxComputeWorkgroupSizeZ                  uint32
	MaxComputeWorkgroupsPerDimension          uint32

	// MaxPushConstantSize maps to v29's core WGPULimits.maxImmediateSize
	// (push constants became "immediates" in v29).
	MaxPushConstantSize uint32
}

// SupportedLimits / RequiredLimits wrap Limits, matching cogentcore's shape.
type SupportedLimits struct{ Limits Limits }
type RequiredLimits struct{ Limits Limits }

// DefaultLimits returns a limit set that is entirely "undefined" — i.e. imposes
// no requirements, so RequestDevice gets the adapter's defaults. Matches
// cogentcore: notably MaxBufferSize stays at the u64 sentinel until explicitly
// set, so a `<` guard never silently keeps a low default.
func DefaultLimits() Limits {
	return Limits{
		MaxTextureDimension1D:                     LimitU32Undefined,
		MaxTextureDimension2D:                     LimitU32Undefined,
		MaxTextureDimension3D:                     LimitU32Undefined,
		MaxTextureArrayLayers:                     LimitU32Undefined,
		MaxBindGroups:                             LimitU32Undefined,
		MaxBindGroupsPlusVertexBuffers:            LimitU32Undefined,
		MaxBindingsPerBindGroup:                   LimitU32Undefined,
		MaxDynamicUniformBuffersPerPipelineLayout: LimitU32Undefined,
		MaxDynamicStorageBuffersPerPipelineLayout: LimitU32Undefined,
		MaxSampledTexturesPerShaderStage:          LimitU32Undefined,
		MaxSamplersPerShaderStage:                 LimitU32Undefined,
		MaxStorageBuffersPerShaderStage:           LimitU32Undefined,
		MaxStorageTexturesPerShaderStage:          LimitU32Undefined,
		MaxUniformBuffersPerShaderStage:           LimitU32Undefined,
		MaxUniformBufferBindingSize:               LimitU64Undefined,
		MaxStorageBufferBindingSize:               LimitU64Undefined,
		MinUniformBufferOffsetAlignment:           LimitU32Undefined,
		MinStorageBufferOffsetAlignment:           LimitU32Undefined,
		MaxVertexBuffers:                          LimitU32Undefined,
		MaxBufferSize:                             LimitU64Undefined,
		MaxVertexAttributes:                       LimitU32Undefined,
		MaxVertexBufferArrayStride:                LimitU32Undefined,
		MaxInterStageShaderComponents:             LimitU32Undefined,
		MaxInterStageShaderVariables:              LimitU32Undefined,
		MaxColorAttachments:                       LimitU32Undefined,
		MaxColorAttachmentBytesPerSample:          LimitU32Undefined,
		MaxComputeWorkgroupStorageSize:            LimitU32Undefined,
		MaxComputeInvocationsPerWorkgroup:         LimitU32Undefined,
		MaxComputeWorkgroupSizeX:                  LimitU32Undefined,
		MaxComputeWorkgroupSizeY:                  LimitU32Undefined,
		MaxComputeWorkgroupSizeZ:                  LimitU32Undefined,
		MaxComputeWorkgroupsPerDimension:          LimitU32Undefined,
		MaxPushConstantSize:                       LimitU32Undefined,
	}
}

// limitsFromC reads a populated C.WGPULimits into a Go Limits.
func limitsFromC(l C.WGPULimits) Limits {
	return Limits{
		MaxTextureDimension1D:                     uint32(l.maxTextureDimension1D),
		MaxTextureDimension2D:                     uint32(l.maxTextureDimension2D),
		MaxTextureDimension3D:                     uint32(l.maxTextureDimension3D),
		MaxTextureArrayLayers:                     uint32(l.maxTextureArrayLayers),
		MaxBindGroups:                             uint32(l.maxBindGroups),
		MaxBindGroupsPlusVertexBuffers:            uint32(l.maxBindGroupsPlusVertexBuffers),
		MaxBindingsPerBindGroup:                   uint32(l.maxBindingsPerBindGroup),
		MaxDynamicUniformBuffersPerPipelineLayout: uint32(l.maxDynamicUniformBuffersPerPipelineLayout),
		MaxDynamicStorageBuffersPerPipelineLayout: uint32(l.maxDynamicStorageBuffersPerPipelineLayout),
		MaxSampledTexturesPerShaderStage:          uint32(l.maxSampledTexturesPerShaderStage),
		MaxSamplersPerShaderStage:                 uint32(l.maxSamplersPerShaderStage),
		MaxStorageBuffersPerShaderStage:           uint32(l.maxStorageBuffersPerShaderStage),
		MaxStorageTexturesPerShaderStage:          uint32(l.maxStorageTexturesPerShaderStage),
		MaxUniformBuffersPerShaderStage:           uint32(l.maxUniformBuffersPerShaderStage),
		MaxUniformBufferBindingSize:               uint64(l.maxUniformBufferBindingSize),
		MaxStorageBufferBindingSize:               uint64(l.maxStorageBufferBindingSize),
		MinUniformBufferOffsetAlignment:           uint32(l.minUniformBufferOffsetAlignment),
		MinStorageBufferOffsetAlignment:           uint32(l.minStorageBufferOffsetAlignment),
		MaxVertexBuffers:                          uint32(l.maxVertexBuffers),
		MaxBufferSize:                             uint64(l.maxBufferSize),
		MaxVertexAttributes:                       uint32(l.maxVertexAttributes),
		MaxVertexBufferArrayStride:                uint32(l.maxVertexBufferArrayStride),
		MaxInterStageShaderVariables:              uint32(l.maxInterStageShaderVariables),
		MaxColorAttachments:                       uint32(l.maxColorAttachments),
		MaxColorAttachmentBytesPerSample:          uint32(l.maxColorAttachmentBytesPerSample),
		MaxComputeWorkgroupStorageSize:            uint32(l.maxComputeWorkgroupStorageSize),
		MaxComputeInvocationsPerWorkgroup:         uint32(l.maxComputeInvocationsPerWorkgroup),
		MaxComputeWorkgroupSizeX:                  uint32(l.maxComputeWorkgroupSizeX),
		MaxComputeWorkgroupSizeY:                  uint32(l.maxComputeWorkgroupSizeY),
		MaxComputeWorkgroupSizeZ:                  uint32(l.maxComputeWorkgroupSizeZ),
		MaxComputeWorkgroupsPerDimension:          uint32(l.maxComputeWorkgroupsPerDimension),
		MaxPushConstantSize:                       uint32(l.maxImmediateSize),
	}
}

// fillCLimits writes a Go Limits into a (zeroed) C.WGPULimits. Fields with no
// Go counterpart are left at the undefined sentinel ("no requirement").
func fillCLimits(c *C.WGPULimits, l Limits) {
	c.maxTextureDimension1D = C.uint32_t(l.MaxTextureDimension1D)
	c.maxTextureDimension2D = C.uint32_t(l.MaxTextureDimension2D)
	c.maxTextureDimension3D = C.uint32_t(l.MaxTextureDimension3D)
	c.maxTextureArrayLayers = C.uint32_t(l.MaxTextureArrayLayers)
	c.maxBindGroups = C.uint32_t(l.MaxBindGroups)
	if l.MaxBindGroupsPlusVertexBuffers != 0 {
		c.maxBindGroupsPlusVertexBuffers = C.uint32_t(l.MaxBindGroupsPlusVertexBuffers)
	} else {
		c.maxBindGroupsPlusVertexBuffers = C.uint32_t(LimitU32Undefined)
	}
	c.maxBindingsPerBindGroup = C.uint32_t(l.MaxBindingsPerBindGroup)
	c.maxDynamicUniformBuffersPerPipelineLayout = C.uint32_t(l.MaxDynamicUniformBuffersPerPipelineLayout)
	c.maxDynamicStorageBuffersPerPipelineLayout = C.uint32_t(l.MaxDynamicStorageBuffersPerPipelineLayout)
	c.maxSampledTexturesPerShaderStage = C.uint32_t(l.MaxSampledTexturesPerShaderStage)
	c.maxSamplersPerShaderStage = C.uint32_t(l.MaxSamplersPerShaderStage)
	c.maxStorageBuffersPerShaderStage = C.uint32_t(l.MaxStorageBuffersPerShaderStage)
	c.maxStorageTexturesPerShaderStage = C.uint32_t(l.MaxStorageTexturesPerShaderStage)
	c.maxUniformBuffersPerShaderStage = C.uint32_t(l.MaxUniformBuffersPerShaderStage)
	c.maxUniformBufferBindingSize = C.uint64_t(l.MaxUniformBufferBindingSize)
	c.maxStorageBufferBindingSize = C.uint64_t(l.MaxStorageBufferBindingSize)
	c.minUniformBufferOffsetAlignment = C.uint32_t(l.MinUniformBufferOffsetAlignment)
	c.minStorageBufferOffsetAlignment = C.uint32_t(l.MinStorageBufferOffsetAlignment)
	c.maxVertexBuffers = C.uint32_t(l.MaxVertexBuffers)
	c.maxBufferSize = C.uint64_t(l.MaxBufferSize)
	c.maxVertexAttributes = C.uint32_t(l.MaxVertexAttributes)
	c.maxVertexBufferArrayStride = C.uint32_t(l.MaxVertexBufferArrayStride)
	c.maxInterStageShaderVariables = C.uint32_t(l.MaxInterStageShaderVariables)
	c.maxColorAttachments = C.uint32_t(l.MaxColorAttachments)
	c.maxColorAttachmentBytesPerSample = C.uint32_t(l.MaxColorAttachmentBytesPerSample)
	c.maxComputeWorkgroupStorageSize = C.uint32_t(l.MaxComputeWorkgroupStorageSize)
	c.maxComputeInvocationsPerWorkgroup = C.uint32_t(l.MaxComputeInvocationsPerWorkgroup)
	c.maxComputeWorkgroupSizeX = C.uint32_t(l.MaxComputeWorkgroupSizeX)
	c.maxComputeWorkgroupSizeY = C.uint32_t(l.MaxComputeWorkgroupSizeY)
	c.maxComputeWorkgroupSizeZ = C.uint32_t(l.MaxComputeWorkgroupSizeZ)
	c.maxComputeWorkgroupsPerDimension = C.uint32_t(l.MaxComputeWorkgroupsPerDimension)
	c.maxImmediateSize = C.uint32_t(l.MaxPushConstantSize)
}
