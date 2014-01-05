package cuda

import (
	"reflect"
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
