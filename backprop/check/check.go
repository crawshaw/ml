package check

import (
	"github.com/crawshaw/ml/backprop"
)

// Tests is a set of standard inputs with expected outputs for backpropagation.
var Tests = []struct {
	Name        string
	Src         backprop.Backprop
	WantDelta   []float32
	WantDownErr []float32
}{
	{
		Name: "1x1 degenerate fixed point",
		Src: backprop.Backprop{
			Delta:    []float32{0},
			DownErr:  []float32{0},
			UpErr:    []float32{0},
			UpOrig:   []float32{1},
			DownOrig: []float32{1},
			Param:    []float32{1},
		},
		WantDelta:   []float32{0},
		WantDownErr: []float32{0},
	},
}
