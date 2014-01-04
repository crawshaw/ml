package f16

// #cgo CFLAGS: -mavx -mf16c
// #include "f16.h"
import "C"
import "fmt"

type Float16 C.f16

func Encode(dst []Float16, src []float32) {
	if len(src) != len(dst) {
		panic(fmt.Sprintf("nn: len(dst)=%d, not equal to len(src)=%d", len(dst), len(src)))
	}
	C.ftof16((*C.f16)(&dst[0]), (*C.float)(&src[0]), C.int(len(src)))
}

func Decode(dst []float32, src []Float16) {
	if len(src) != len(dst) {
		panic(fmt.Sprintf("nn: len(dst)=%d, not equal to len(src)=%d", len(dst), len(src)))
	}
	C.f16tof((*C.float)(&dst[0]), (*C.f16)(&src[0]), C.int(len(src)))
}
