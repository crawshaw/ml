package nn

import (
	"reflect"
	"testing"
)

func TestToFloat16(t *testing.T) {
	src := []float32{8, 79, 0, 1, 2, 3, 4, 5}
	f := make([]Float16, len(src))
	Float32ToFloat16(f, src)
	dst := make([]float32, len(src))
	Float16ToFloat32(dst, f)
	if !reflect.DeepEqual(src, dst) {
		t.Errorf("src %v != dst %v", src, dst)
	}
}

func f16(x []float32) []Float16 {
	res := make([]Float16, len(x))
	Float32ToFloat16(res, x)
	return res
}

func TestForward(t *testing.T) {
	l0, l1 := Alloc(8), Alloc(8)
	l0.ToDevice(f16([]float32{2, 2, 2, 2, 2, 2, 2, 2}))
	l1.ToDevice(f16([]float32{0, 0, 0, 0, 0, 0, 0, 0}))

	p := Alloc(8*8)
	p.ToDevice(f16([]float32{
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

	res16 := make([]Float16, 8)
	res32 := make([]float32, 8)
	l1.ToHost(res16)
	Float16ToFloat32(res32, res16)

	want := []float32{6, 8, 6, 4, 4, 8, 8, 2}
	if !reflect.DeepEqual(want, res32) {
		t.Errorf("got %v != want %v", res32, want)
	}
}
