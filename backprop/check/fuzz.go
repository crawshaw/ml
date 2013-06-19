// Package check provides standard tests for the backprop package
// and tools for fuzz comparison between different backprop
// implementations.
package check

import (
	"math/rand"
	"testing"

	"github.com/crawshaw/ml/backprop"
)

// Fuzz generates random belief networks and compares the given
// backpropagation implementation against the reference implementation.
func Fuzz(t *testing.T, f func(*backprop.Backprop)) {
	// Use deterministic random source so our fuzz testing is not flaky.
	rnd := rand.New(rand.NewSource(42))

	const (
		runs = 1000
		maxLayerSize = 100
	)
	failures := 0
	for run := 0; run < runs; run++ {
		numDown, numUp := rnd.Intn(maxLayerSize)+1, rnd.Intn(maxLayerSize)+1
		b1 := &backprop.Backprop{
			UpOrig:   make([]float32, numUp),
			UpErr:    make([]float32, numUp),
			DownOrig: make([]float32, numDown),
			DownErr:  make([]float32, numDown),
			Param:    make([]float32, numDown*numUp),
			Delta:    make([]float32, numDown*numUp),
		}
		for i := 0; i < numDown*numUp; i++ {
			b1.Param[i] = rnd.Float32()
		}
		for i := 0; i < numDown; i++ {
			b1.DownOrig[i] = rnd.Float32()
		}
		backprop.Forward(b1.UpOrig, b1.DownOrig, b1.Param)
		for i := 0; i < numUp; i++ {
			b1.UpErr[i] = rnd.Float32()
		}
		b2 := deepcopy(b1)
		b1.Backward()
		f(b2)

		if !eq(b1.Delta, b2.Delta) || !eq(b1.DownErr, b2.DownErr) {
			failures++
			if failures > 10 {
				continue
			}
			t.Errorf("%d: got\n\tDelta:   %v\n\tDownErr: %v\nwant\n\tDelta:   %v\n\tDownErr: %v", run, b2.Delta, b2.DownErr, b1.Delta , b1.DownErr)
		}
	}
	if failures > 0 {
		t.Errorf("%d/%d fuzz test failures", failures, runs)
	}
}

func deepcopy(b1 *backprop.Backprop) *backprop.Backprop {
	numUp, numDown := len(b1.UpOrig), len(b1.DownOrig)
	b2 := &backprop.Backprop{
		UpOrig:   make([]float32, numUp),
		UpErr:    make([]float32, numUp),
		DownOrig: make([]float32, numDown),
		DownErr:  make([]float32, numDown),
		Param:    make([]float32, numDown*numUp),
		Delta:    make([]float32, numDown*numUp),
	}
	copy(b2.UpOrig, b1.UpOrig)
	copy(b2.UpErr, b1.UpErr)
	copy(b2.DownOrig, b1.DownOrig)
	copy(b2.Param, b1.Param)
	return b2
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func eq(x, y []float32) bool {
	if len(x) != len(y) {
		return false
	}
	for i := 0; i < len(x); i++ {
		const ε = 1e-6
		if abs(x[i]-y[i]) > ε {
			return false
		}
	}
	return true
}
