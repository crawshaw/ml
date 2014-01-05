package f16

import (
	"reflect"
	"testing"
)

func TestEncodeDecode(t *testing.T) {
	src := make([]float32, 128)
	for i := range src {
		src[i] = float32(i)
	}
	src16 := make([]Float16, len(src))
	Encode(src16, src)
	dst := make([]float32, len(src))
	Decode(dst, src16)
	if !reflect.DeepEqual(src, dst) {
		t.Errorf("src %v != dst %v", src, dst)
	}
}

func TestAdd(t *testing.T) {
	wrk := mkf16([]float32{0, 1, 2, 3, 4, 5, 6, 7})
	toadd := mkf16([]float32{1, 2, 2, 2, 2, 2, 2, 1})
	want := []float32{1, 3, 4, 5, 6, 7, 8, 8}

	Add(wrk, toadd)

	got := fromf16(wrk)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("got %v != want %v", got, want)
	}
}

func TestSub(t *testing.T) {
	wrk := mkf16([]float32{1, 3, 4, 5, 6, 7, 8, 8})
	tosub := mkf16([]float32{1, 2, 2, 2, 2, 2, 2, 1})
	want := []float32{0, 1, 2, 3, 4, 5, 6, 7}

	Sub(wrk, tosub)

	got := fromf16(wrk)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("got %v != want %v", got, want)
	}
}

func TestMul(t *testing.T) {
	wrk := mkf16([]float32{1, 3, 0.5, 4, 5, 7, 8, 8})
	want := []float32{2, 6, 1, 8, 10, 14, 16, 16}

	Mul(wrk, 2)

	got := fromf16(wrk)

	if !reflect.DeepEqual(want, got) {
		t.Errorf("got %v != want %v", got, want)
	}
}

func BenchmarkAdd(b *testing.B) {
	b.StopTimer()
	src1, src2 := mkAddSrcs()
	res := make([]float32, len(src1))

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copy(res, src1)
		b.StartTimer()
		for i := range res {
			res[i] += src2[i]
		}
	}
}

func BenchmarkAdd16(b *testing.B) {
	b.StopTimer()
	s1, s2 := mkAddSrcs()
	src1, src2 := mkf16(s1), mkf16(s2)
	res := make([]Float16, len(src1))

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		copy(res, src1)
		b.StartTimer()
		Add(res, src2)
	}
}

func mkAddSrcs() ([]float32, []float32) {
	const size = 1e5
	src1, src2 := make([]float32, size), make([]float32, size)
	for i := range src1 {
		src1[i] = float32(i%11)
		src2[i] = float32(i%15)
	}
	return src1, src2
}

func mkf16(x []float32) []Float16 {
	res := make([]Float16, len(x))
	Encode(res, x)
	return res
}

func fromf16(x []Float16) []float32 {
	res := make([]float32, len(x))
	Decode(res, x)
	return res
}
