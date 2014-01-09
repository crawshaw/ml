package cuda

import (
	"reflect"
	"runtime"
	"testing"

	"github.com/crawshaw/ml/f16"
)

func TestDeviceCopy(t *testing.T) {
	want := []float32{1, 3, 4, 5, 6, 7, 8, 2}
	d := Alloc(len(want))
	d.ToDevice(mkf16(want))

	got16 := make([]f16.Float16, len(want))
	got := make([]float32, len(want))
	d.ToHost(got16)
	f16.Decode(got, got16)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v != want %v", got, want)
	}
}

func mkf16(x []float32) []f16.Float16 {
	res := make([]f16.Float16, len(x))
	f16.Encode(res, x)
	return res
}

func TestSub(t *testing.T) {
	dst, l0, l1 := Alloc(8), Alloc(8), Alloc(8)
	l0.ToDevice(mkf16([]float32{8, 7, 2, 4, 2, 4, 2, 1}))
	l1.ToDevice(mkf16([]float32{6, 1, 4, 4, 0, 1, 2, 8}))
	dst.ToDevice(mkf16([]float32{0, 0, 0, 0, 0, 0, 0, 0}))
	want := []float32{2, 6, -2, 0, 2, 3, 0, -7}
	Sub(dst, l0, l1)

	res := make([]f16.Float16, len(want))
	dst.ToHost(res)
	got := make([]float32, len(want))
	f16.Decode(got, res)

	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v != want %v", got, want)
	}
}

func TestFree(t *testing.T) {
	// not a great test, mostly checks for any unexpected panic.
	_ = Alloc(1<<20)
	runtime.GC()
}

func TestForward(t *testing.T) {
	l0, l1 := Alloc(8), Alloc(8)
	l0.ToDevice(mkf16([]float32{2, 2, 2, 2, 2, 2, 2, 2}))
	l1.ToDevice(mkf16([]float32{0, 0, 0, 0, 0, 0, 0, 0}))

	p := Alloc(8 * 8)
	p.ToDevice(mkf16([]float32{
		1, 2, 0, 0, 0, 0, 0, 0,
		0, 4, 0, 0, 0, 0, 0, 0,
		0, 0, 3, 0, 0, 0, 0, 0,
		0, 0, 0, 2, 0, 0, 0, 0,
		0, 0, 0, 0, 2, 0, 0, 0,
		0, 0, 0, 0, 1, 3, 0, 0,
		0, 0, 0, 0, 0, 0, 4, 0,
		0, 0, 0, 0, 0, 0, 0, 1,
	}))

	Forward(l1, l0, p)

	res16 := make([]f16.Float16, 8)
	res32 := make([]float32, 8)
	l1.ToHost(res16)
	f16.Decode(res32, res16)

	want := []float32{6, 8, 6, 4, 4, 8, 8, 2}
	if !reflect.DeepEqual(want, res32) {
		t.Errorf("got %v != want %v", res32, want)
	}
}
