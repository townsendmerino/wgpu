package wgpu

import "unsafe"

// FromBytes reinterprets a byte slice as a slice of E (no copy). Mirrors
// cogentcore/webgpu's helper of the same name. Panics if len is not a multiple
// of the element size.
func FromBytes[E any](src []byte) []E {
	l := uintptr(len(src))
	if l == 0 {
		return nil
	}
	var zero E
	elmSize := unsafe.Sizeof(zero)
	if l%elmSize != 0 {
		panic("wgpu.FromBytes: byte length not a multiple of element size")
	}
	return unsafe.Slice((*E)(unsafe.Pointer(&src[0])), l/elmSize)
}

// ToBytes reinterprets a slice of E as a byte slice (no copy).
func ToBytes[E any](src []E) []byte {
	l := uintptr(len(src))
	if l == 0 {
		return nil
	}
	elmSize := unsafe.Sizeof(src[0])
	return unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), l*elmSize)
}
