// Package f16 supports half-precision floating point numbers.
//
// Half-precision floats should be considered a storage mechanism
// only. Arithmetic is peformed by converting to 32-bit floats.
//
// All functions in this package operate over ranges of Float16
// values to take advantage of the F16C instructions in SSE5:
//
//	http://en.wikipedia.org/wiki/F16C
package f16

// #cgo CFLAGS: -mavx -mf16c
// #include "f16.h"
import "C"
import "fmt"

// Float16 is a 16-bit "half-precision" floating point number.
type Float16 C.f16

// Encode encodes a slice of float32 values into a Float16 slice.
// Panics if the slices have different lengths.
func Encode(dst []Float16, src []float32) {
	if msg := sameLen(len(dst), len(src)); msg != "" {
		panic(msg)
	}
	C.ftof16((*C.f16)(&dst[0]), (*C.float)(&src[0]), C.int(len(src)))
}

// Decode decodes a slice of Float16 values into a float32 slice.
// Panics if the slices have different lengths.
func Decode(dst []float32, src []Float16) {
	if msg := sameLen(len(dst), len(src)); msg != "" {
		panic(msg)
	}
	C.f16tof((*C.float)(&dst[0]), (*C.f16)(&src[0]), C.int(len(src)))
}

// Add adds each value of src to the corresponding position in dst.
// Panics if the slices have different lengths.
func Add(dst, src []Float16) {
	if msg := sameLen(len(dst), len(src)); msg != "" {
		panic(msg)
	}
	C.f16add((*C.f16)(&dst[0]), (*C.f16)(&src[0]), C.int(len(src)))
}

// Sub subtracts each value of src to the corresponding position in dst.
// Panics if the slices have different lengths.
func Sub(dst, src []Float16) {
	if msg := sameLen(len(dst), len(src)); msg != "" {
		panic(msg)
	}
	C.f16sub((*C.f16)(&dst[0]), (*C.f16)(&src[0]), C.int(len(src)))
}

// Mul multiples each value of dst with the given value.
func Mul(dst []Float16, val float32) {
	C.f16mul((*C.f16)(&dst[0]), C.int(len(dst)), C.float(val))
}

func sameLen(dst, src int) string {
	if dst != src {
		return fmt.Sprintf("f16: len(dst)=%d, not equal to len(src)=%d", dst, src)
	}
	return ""
}
