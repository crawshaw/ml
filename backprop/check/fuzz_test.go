package check

import (
	"testing"

	"github.com/crawshaw/ml/backprop"
)

func TestFuzzIdentity(t *testing.T) {
	// fuzz against itself, should always match
	Fuzz(t, func(b *backprop.Backprop) { b.Backward() })
}
