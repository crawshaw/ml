package check

import (
	"github.com/crawshaw/ml/backprop"
)

// Tests is a set of standard inputs with expected outputs for backpropagation.
var Tests = []struct{
	src  backprop.Backprop
	want backprop.Backprop
}{
	{},
}
