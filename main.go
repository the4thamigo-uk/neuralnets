package main

import (
	_ "errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
)

const defaultWeight = 1.0

type (
	net struct {
		dim    int
		layers []*layer
	}

	layer struct {
		weights    *mat.Dense
		activation func(float64) float64
	}
)

func main() {
	err := run()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func run() error {
	n := newNet(2, 4, 1)
	in := mat.NewVecDense(2, []float64{1.0, 2.0})
	fmt.Println(n.calculate(in))
	return nil
}

func newNet(dims ...int) *net {
	var lrs []*layer
	for i := 0; i < len(dims)-1; i++ {
		r := dims[i+1]
		c := dims[i]
		w := mat.NewDense(r, c, makeSlice(r*c, defaultWeight))
		lrs = append(lrs, &layer{weights: w})
	}
	return &net{
		layers: lrs,
	}
}

func (n *net) calculate(in mat.Vector) mat.Vector {
	for _, l := range n.layers {
		var out mat.Dense
		out.Mul(l.weights, in)
		out.Apply(applySigmoid, &out)
		in = out.ColView(0)
	}
	return in
}

func makeSlice(n int, val float64) []float64 {
	s := make([]float64, n, n)
	for i := range s {
		s[i] = val
	}
	return s
}

func rawColView(d *mat.Dense, col int) []float64 {
	rows, _ := d.Dims()
	results := mat.NewVecDense(rows, nil)
	results.CopyVec(d.ColView(col))
	return results.RawVector().Data
}

func applySigmoid(i, j int, v float64) float64 {
	return sigmoid(v)
}

func sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}
