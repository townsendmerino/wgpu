//go:build cgo

package wgpu

/*
#include "./lib/wgpu.h"
*/
import "C"

// BufferUsage is a bitset of WGPUBufferUsage flags (64-bit in v29).
type BufferUsage uint64

const (
	BufferUsageNone    BufferUsage = C.WGPUBufferUsage_None
	BufferUsageMapRead BufferUsage = C.WGPUBufferUsage_MapRead
	BufferUsageCopySrc BufferUsage = C.WGPUBufferUsage_CopySrc
	BufferUsageCopyDst BufferUsage = C.WGPUBufferUsage_CopyDst
	BufferUsageUniform BufferUsage = C.WGPUBufferUsage_Uniform
	BufferUsageStorage BufferUsage = C.WGPUBufferUsage_Storage
	// QueryResolve is needed for the destination of ResolveQuerySet (timestamps).
	BufferUsageQueryResolve BufferUsage = C.WGPUBufferUsage_QueryResolve
)

// MapMode is a bitset of WGPUMapMode flags.
type MapMode uint64

const (
	MapModeNone  MapMode = C.WGPUMapMode_None
	MapModeRead  MapMode = C.WGPUMapMode_Read
	MapModeWrite MapMode = C.WGPUMapMode_Write
)

// PowerPreference mirrors WGPUPowerPreference.
type PowerPreference uint32

const (
	PowerPreferenceUndefined       PowerPreference = C.WGPUPowerPreference_Undefined
	PowerPreferenceLowPower        PowerPreference = C.WGPUPowerPreference_LowPower
	PowerPreferenceHighPerformance PowerPreference = C.WGPUPowerPreference_HighPerformance
)

// FeatureName mirrors WGPUFeatureName (standard webgpu features).
type FeatureName uint32

const (
	FeatureNameTimestampQuery    FeatureName = C.WGPUFeatureName_TimestampQuery
	FeatureNameShaderF16         FeatureName = C.WGPUFeatureName_ShaderF16
	FeatureNameFloat32Filterable FeatureName = C.WGPUFeatureName_Float32Filterable
	FeatureNameSubgroups         FeatureName = C.WGPUFeatureName_Subgroups
)

// AdapterType mirrors WGPUAdapterType.
type AdapterType uint32

const (
	AdapterTypeDiscreteGPU   AdapterType = C.WGPUAdapterType_DiscreteGPU
	AdapterTypeIntegratedGPU AdapterType = C.WGPUAdapterType_IntegratedGPU
	AdapterTypeCPU           AdapterType = C.WGPUAdapterType_CPU
	AdapterTypeUnknown       AdapterType = C.WGPUAdapterType_Unknown
)

func (t AdapterType) String() string {
	switch t {
	case AdapterTypeDiscreteGPU:
		return "DiscreteGPU"
	case AdapterTypeIntegratedGPU:
		return "IntegratedGPU"
	case AdapterTypeCPU:
		return "CPU"
	default:
		return "Unknown"
	}
}

// BackendType mirrors WGPUBackendType.
type BackendType uint32

const (
	BackendTypeUndefined BackendType = C.WGPUBackendType_Undefined
	BackendTypeNull      BackendType = C.WGPUBackendType_Null
	BackendTypeWebGPU    BackendType = C.WGPUBackendType_WebGPU
	BackendTypeD3D11     BackendType = C.WGPUBackendType_D3D11
	BackendTypeD3D12     BackendType = C.WGPUBackendType_D3D12
	BackendTypeMetal     BackendType = C.WGPUBackendType_Metal
	BackendTypeVulkan    BackendType = C.WGPUBackendType_Vulkan
	BackendTypeOpenGL    BackendType = C.WGPUBackendType_OpenGL
	BackendTypeOpenGLES  BackendType = C.WGPUBackendType_OpenGLES
)

func (t BackendType) String() string {
	switch t {
	case BackendTypeNull:
		return "Null"
	case BackendTypeWebGPU:
		return "WebGPU"
	case BackendTypeD3D11:
		return "D3D11"
	case BackendTypeD3D12:
		return "D3D12"
	case BackendTypeMetal:
		return "Metal"
	case BackendTypeVulkan:
		return "Vulkan"
	case BackendTypeOpenGL:
		return "OpenGL"
	case BackendTypeOpenGLES:
		return "OpenGLES"
	default:
		return "Undefined"
	}
}

// BufferMapAsyncStatus mirrors the relevant WGPUMapAsyncStatus values. The
// names match cogentcore/webgpu for drop-in compatibility. Unknown (0) is the
// zero value (the in-flight sentinel callers initialise to).
type BufferMapAsyncStatus uint32

const (
	BufferMapAsyncStatusUnknown           BufferMapAsyncStatus = 0
	BufferMapAsyncStatusSuccess           BufferMapAsyncStatus = C.WGPUMapAsyncStatus_Success
	BufferMapAsyncStatusCallbackCancelled BufferMapAsyncStatus = C.WGPUMapAsyncStatus_CallbackCancelled
	BufferMapAsyncStatusError             BufferMapAsyncStatus = C.WGPUMapAsyncStatus_Error
	BufferMapAsyncStatusAborted           BufferMapAsyncStatus = C.WGPUMapAsyncStatus_Aborted
)

func (s BufferMapAsyncStatus) String() string {
	switch s {
	case BufferMapAsyncStatusSuccess:
		return "Success"
	case BufferMapAsyncStatusCallbackCancelled:
		return "CallbackCancelled"
	case BufferMapAsyncStatusError:
		return "Error"
	case BufferMapAsyncStatusAborted:
		return "Aborted"
	default:
		return "Unknown"
	}
}

// QueryType mirrors WGPUQueryType.
type QueryType uint32

const (
	QueryTypeOcclusion QueryType = C.WGPUQueryType_Occlusion
	QueryTypeTimestamp QueryType = C.WGPUQueryType_Timestamp
)

// ErrorType mirrors WGPUErrorType.
type ErrorType uint32

const (
	ErrorTypeNoError     ErrorType = C.WGPUErrorType_NoError
	ErrorTypeValidation  ErrorType = C.WGPUErrorType_Validation
	ErrorTypeOutOfMemory ErrorType = C.WGPUErrorType_OutOfMemory
	ErrorTypeInternal    ErrorType = C.WGPUErrorType_Internal
	ErrorTypeUnknown     ErrorType = C.WGPUErrorType_Unknown
)

// NativeFeature mirrors wgpu-native's WGPUNativeFeature extension enum. These
// sit in the same WGPUFeatureName value space (0x0003xxxx) and can be passed in
// DeviceDescriptor.RequiredFeatures and queried via HasFeature.
type NativeFeature uint32

const (
	NativeFeatureImmediates                 NativeFeature = C.WGPUNativeFeature_Immediates
	NativeFeaturePipelineStatisticsQuery    NativeFeature = C.WGPUNativeFeature_PipelineStatisticsQuery
	NativeFeatureSubgroup                   NativeFeature = C.WGPUNativeFeature_Subgroup
	NativeFeatureSubgroupBarrier            NativeFeature = C.WGPUNativeFeature_SubgroupBarrier
	NativeFeatureTimestampQueryInsidePasses NativeFeature = C.WGPUNativeFeature_TimestampQueryInsidePasses
	NativeFeatureShaderInt64                NativeFeature = C.WGPUNativeFeature_ShaderInt64
)

// AsFeatureName lets a NativeFeature be used wherever a FeatureName is expected.
func (f NativeFeature) AsFeatureName() FeatureName { return FeatureName(f) }
