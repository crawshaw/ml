// Package backprop implements backpropagation, a supervised learning method
// for training belief systems (neural networks).
//
// This package only implements the step between two layers. (That alone is
// notoriously difficult to debug and test.) For general purpose gradient
// descent over a neural network, see http://github.com/crawshaw/ml/nn.
//
// Consider this package a reference implementation for faster implementations.
//
// Background reading:
//	http://www.stanford.edu/class/cs294a/sparseAutoencoder.pdf
//	http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
package backprop

import (
	"fmt"
	"math"
)

// Forward implements forward propagation for a single fully connected layer.
//	i                      -> i + 1
//	sigmoid(downVal*param) -> upVal
func Forward(upVal, downVal, param []float32) {
	for j := range upVal {
		d := float32(0)
		for i, s := range downVal {
			p := param[i*len(upVal)+j]
			d += s * p
		}
		upVal[j] = sigmoid(d)
	}
}

// Backprop is the arguments for backpropagation.
//
// Key array indicies: DownErr[i] <- Param[i*len(UpErr)+j] <- UpErr[j]
type Backprop struct {
	// Output
	Delta    []float32 // parameter changes
	DownErr  []float32 // error in the lower layer neuron values

	// Input
	UpErr    []float32 // error in the upper layer neuron values
	UpOrig   []float32 // original neuron values
	DownOrig []float32 // original neuron values
	Param    []float32 // "synaptic weights"
}

// Backward implements backpropagation for a single fully connected layer.
//	downErr <- sum(param*upErr)
//	delta 	<- upErr*downOrig*sigmoidGradient(upOrig)
func (b *Backprop) Backward() {
	if len(b.DownErr) != len(b.Param)/len(b.UpErr) {
		panic(fmt.Sprintf("layers don't line up: len(downErr)=%d, len(param)=%d, len(upErr)=%d",
			len(b.DownErr), len(b.Param), len(b.UpErr)))
	}

	// Calculate next error round.
	for i := range b.DownErr {
		d := float32(0)
		for j := range b.UpErr {
			p := b.Param[i*len(b.UpErr)+j]
			d += p * b.UpErr[j]
		}
		b.DownErr[i] = d
	}

	// Calculate parameter delta.
	for j := range b.UpErr {
		grad := sigmoidGradient(b.UpOrig[j])
		for i := range b.DownOrig {
			b.Delta[i*len(b.UpErr)+j] += b.UpErr[j] * grad * b.DownOrig[i]
		}
	}
}

func sigmoid(x float32) float32 {
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func sigmoidGradient(x float32) float32 {
	return x * (1 - x)
}
