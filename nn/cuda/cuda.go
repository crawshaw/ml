package cuda

/*
#cgo LDFLAGS: cudann.so -lcudart
#include "cudann.h"
*/
import "C"
import (
	"fmt"
	"runtime"

	"github.com/crawshaw/ml/f16"
)

// cudaf16mem is designed to be used as a Go pointer responsible for owning
// a piece of allocated CUDA memory. When cudaf16mem is cleaned up by the
// garbage collector, the associated memory is deallocated.
type cudaf16mem struct {
	d *C.f16
}

// Float16Device is an array of float16 values stored on the CUDA device.
// The only operations that can be performed on this memory are load/store
// from host memory, and run a CUDA kernel function.
type Float16Device struct {
	d     *cudaf16mem
	count int
}

func (p Float16Device) Len() int { return p.count }

func (p Float16Device) ToDevice(h []f16.Float16) {
	if len(h) != p.count {
		panic(fmt.Sprintf("Float16Device.ToDevice: len(h)=%d, count=%d", len(h), p.count))
	}
	C.memcpy_htod(p.d.d, (*C.f16)(&h[0]), C.int(p.count))
}

func (p Float16Device) ToHost(h []f16.Float16) {
	if len(h) != p.count {
		panic(fmt.Sprintf("Float16Device.ToHost: len(h)=%d, count=%d", len(h), p.count))
	}
	C.memcpy_dtoh((*C.f16)(&h[0]), p.d.d, C.int(p.count))
}

// Sub subtracts b from a and writes to dst, i.e. dst = a - b.
func Sub(dst, a, b Float16Device) {
	if dst.count != a.count || dst.count != b.count {
		panic(fmt.Sprintf("cuda.Sub: dst.Len=%d, a.Len=%d, b.Len=%d", dst.count, a.count, b.count))
	}
	C.f16devsub(dst.d.d, a.d.d, b.d.d, C.int(dst.count))
}

// TODO: may be easier to read:
// func CopyToHost(dst []f16.Float16, src Float16Device)
// func CopyToDevice(dst Float16Device, src []f16.Float16)

func Alloc(count int) Float16Device {
	d := &cudaf16mem{C.alloc_f16_device(C.int(count))}
	runtime.SetFinalizer(d, func(d *cudaf16mem) {
		C.free_f16_device(d.d)
	})
	return Float16Device{d: d, count: count}
}

func Forward(up, down, param Float16Device) {
	C.forward(up.d.d, C.int(up.count), down.d.d, C.int(down.count), param.d.d)
}

func Backward(up, upErr, down, downErr, param Float16Device) {
	C.backward(up.d.d, upErr.d.d, C.int(up.count), down.d.d, downErr.d.d, C.int(down.count), param.d.d)
}
