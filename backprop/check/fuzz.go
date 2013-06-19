// Package check provides standard tests for the backprop package
// and tools for fuzz comparison between different backprop
// implementations.
package check

import (
	"testing"

	"github.com/crawshaw/ml/backprop"
)

// Fuzz generates random belief networks and compares the given
// backpropagation implementation against the reference implementation.
func Fuzz(t *testing.T, impl func(*backprop.Backprop)) {
}
