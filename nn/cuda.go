package nn

/*
#cgo LDFLAGS: cuda/cudann.so
#include "cuda/cudann.h"
*/
import "C"
import "fmt"

type Float16Device struct {
	d     *C.f16
	count int
}

func (p Float16Device) ToDevice(h []Float16) {
	if len(h) != p.count {
		panic(fmt.Sprintf("Float16Device.ToDevice: len(h)=%d, count=%d", len(h), p.count))
	}
	C.memcpy_htod(p.d, (*C.f16)(&h[0]), C.int(p.count))
}

func (p Float16Device) ToHost(h []Float16) {
	if len(h) != p.count {
		panic(fmt.Sprintf("Float16Device.ToHost: len(h)=%d, count=%d", len(h), p.count))
	}
	C.memcpy_dtoh((*C.f16)(&h[0]), p.d, C.int(p.count))
}

func Alloc(count int) Float16Device {
	return Float16Device{
		d:     C.alloc_f16_device(C.int(count)),
		count: count,
	}
}

type Float16 C.f16

func Float32ToFloat16(dst []Float16, src []float32) {
	if len(src) != len(dst) {
		panic(fmt.Sprintf("nn: len(dst)=%d, not equal to len(src)=%d", len(dst), len(src)))
	}
	C.f32tof16((*C.f16)(&dst[0]), (*C.f32)(&src[0]), C.int(len(src)))
}

func Float16ToFloat32(dst []float32, src []Float16) {
	if len(src) != len(dst) {
		panic(fmt.Sprintf("nn: len(dst)=%d, not equal to len(src)=%d", len(dst), len(src)))
	}
	C.f16tof32((*C.f32)(&dst[0]), (*C.f16)(&src[0]), C.int(len(src)))
}
