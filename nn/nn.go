// Package nn implements a feed-foward multi-layer deep belief network.
package nn

import (
	"fmt"
	"sync"

	"github.com/crawshaw/ml/f16"
	"github.com/crawshaw/ml/nn/cuda"
)

type Layer struct {
	node  cuda.Float16Device
	err   cuda.Float16Device
	param cuda.Float16Device

	host layerHost
}

type layerHost struct {
	param []f16.Float16
	delta []f16.Float16
}

func NewLayer(size int) Layer {
	return Layer{node: cuda.Alloc(size)}
}

type Network struct {
	Layer []Layer // 0 is input layer, len(Layer)-1 is output layer

	// host memory copies of training batch input/output
	batch []f16label
}

// TODO: GPU experimentation needed: is it fine to
// issue a large set of sequential (or concurrent) memcpys, or
// do we need to carefully pack the memory and do it once?

type f16label struct {
	in  cuda.Float16Device
	out cuda.Float16Device
	buf []f16.Float16 // max(in.Len(), out.Len()) for host encoding
}

// TODO: tune
const batchSize = 256

func (n *Network) init() {
	if n.batch != nil {
		return
	}
	inSize := n.Layer[0].node.Len()
	outSize := n.Layer[len(n.Layer)-1].node.Len()
	bufSize := inSize
	if outSize > bufSize {
		bufSize = outSize
	}
	n.batch = make([]f16label, batchSize)
	for i := range n.batch {
		n.batch[i].in = cuda.Alloc(inSize)
		n.batch[i].out = cuda.Alloc(outSize)
		n.batch[i].buf = make([]f16.Float16, bufSize)
	}
	for i := 0; i < len(n.Layer); i++ {
		n.Layer[i].err = cuda.Alloc(n.Layer[i].node.Len())
	}
	for i := 0; i < len(n.Layer)-1; i++ { // no parameters for last layer
		nsize := n.Layer[i].node.Len()
		psize := nsize * n.Layer[i+1].node.Len()
		n.Layer[i].param = cuda.Alloc(psize)
		n.Layer[i].host.param = make([]f16.Float16, psize)
		n.Layer[i].host.delta = make([]f16.Float16, psize)
	}
}

func (n *Network) forward() {
	for i := 0; i < len(n.Layer)-1; i++ {
		up := n.Layer[i+1].node
		down := n.Layer[i].node
		param := n.Layer[i].param
		cuda.Forward(up, down, param)
	}
}

func (n *Network) backward() {
	for i := len(n.Layer) - 1; i > 0; i++ {
		up := n.Layer[i+1].node
		upErr := n.Layer[i+1].err
		down := n.Layer[i].node
		downErr := n.Layer[i].err
		param := n.Layer[i].param
		cuda.Backward(up, upErr, down, downErr, param)
	}
}

func (n *Network) Train(batch []LabelledData) {
	if len(batch) != len(n.batch) {
		panic(fmt.Sprintf("nn: Train requires batch size %d, got %d", len(n.batch), len(batch)))
	}

	// Load work batch onto device.
	var wg sync.WaitGroup
	wg.Add(len(batch))
	for i, b := range batch {
		go func(i int, d LabelledData) {
			buf := n.batch[i].buf[:n.batch[i].in.Len()]
			f16.Encode(buf, d.Input)
			n.batch[i].in.ToDevice(buf)

			buf = n.batch[i].buf[:n.batch[i].out.Len()]
			f16.Encode(buf, d.Output)
			n.batch[i].out.ToDevice(buf)
			wg.Done()
		}(i, b)
	}
	wg.Wait()

	// Run the batch.
	for _, d := range n.batch {
		// Swap in initial values.
		old := n.Layer[0].node
		n.Layer[0].node = d.in
		d.in = old

		n.forward()

		// Calculate error in last layer.
		outLayer := n.Layer[len(n.Layer)-1]
		cuda.Sub(outLayer.err, outLayer.node, d.out)

		n.backward()
	}

	// TODO copy the parameters back out to the host.
}

type LabelledData struct {
	Input  []float32
	Output []float32
}
