module github.com/townsendmerino/wgpu

// cgo is the whole point: cgo callbacks are robust across Go versions, unlike
// the zero-CGO goffi path (go-webgpu) which targets a fixed crosscall2 ABI and
// SIGABRTs at RequestAdapter on Go 1.26.
go 1.26
