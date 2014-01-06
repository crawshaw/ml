package cuda

/*
#cgo LDFLAGS: cudann.so -lcudart
#include "cudann.h"
*/
import "C"
import (
	"fmt"

	"github.com/crawshaw/ml/f16"
)

type Float16Device struct {
	d     *C.f16
	count int
}

func (p Float16Device) Len() int { return p.count }

func (p Float16Device) ToDevice(h []f16.Float16) {
	if len(h) != p.count {
		panic(fmt.Sprintf("Float16Device.ToDevice: len(h)=%d, count=%d", len(h), p.count))
	}
	C.memcpy_htod(p.d, (*C.f16)(&h[0]), C.int(p.count))
}

func (p Float16Device) ToHost(h []f16.Float16) {
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

func Forward(up, down, param Float16Device) {
	C.forward(up.d, C.int(up.count), down.d, C.int(down.count), param.d)
}

func Backward(up, upErr, down, downErr, param Float16Device) {
	C.backward(up.d, upErr.d, C.int(up.count), down.d, downErr.d, C.int(down.count), param.d)
}
