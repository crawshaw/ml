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
