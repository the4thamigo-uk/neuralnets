package main

import (
	"fmt"
	"github.com/davecgh/go-spew/spew"
	"gonum.org/v1/gonum/mat"
	"math"
	"os"
)

const (
	defaultWeight = 1.0
	learningRate  = 0.1
)

type (
	net struct {
		dim    int
		layers []*layer
	}

	layer struct {
		w          *mat.Dense
		activation func(float64) float64
	}

	result struct {
		in   mat.Vector
		out  mat.Vector
		sum  mat.Vector // weighted sum of inputs
		dsig mat.Vector
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

	n.layers[0].w = mat.NewDense(3, 2, []float64{0.8, 0.2, 0.4, 0.9, 0.3, 0.5})
	n.layers[1].w = mat.NewDense(1, 3, []float64{0.3, 0.5, 0.9})

	in := mat.NewVecDense(2, []float64{0.5, 0.5})
	exp := mat.NewVecDense(1, []float64{0.25})

	for i := 0; i < 1000; i++ {
		fmt.Printf("epoch %d\n", i)
		n.learn(in, exp)
	}
	return nil
}

func newNet(dims ...int) *net {
	var lrs []*layer
	for i := 0; i < len(dims)-1; i++ {
		r := dims[i+1]
		c := dims[i]
		w := mat.NewDense(r, c, makeSlice(r*c, defaultWeight))
		lrs = append(lrs, &layer{w: w})
	}
	return &net{
		layers: lrs,
	}
}

func (n *net) learn(in mat.Vector, exp mat.Vector) []*result {
	rr := n.calculate(in)

	lst := rr[len(rr)-1]
	out := lst.out

	spew.Printf("out: %v\n", lst.out)

	var diff mat.Dense
	diff.Sub(exp, out)

	spew.Printf("diff: %v\n", diff)

	var mse, cost, dcost mat.Dense
	mse.MulElem(&diff, &diff)
	N := float64(out.Len())
	cost.Scale(1.0/N, &mse)
	dcost.Scale(2.0/N, &diff)

	var lastDel mat.Matrix

	for i := len(rr) - 1; i >= 0; i-- {
		r := rr[i]
		l := n.layers[i]

		var del mat.Dense
		if i == len(rr)-1 {
			del.MulElem(&dcost, lst.dsig)
		} else {
			nl := n.layers[i+1]
			var wdel mat.Dense
			wdel.Mul(nl.w.T(), lastDel)
			del.MulElem(&wdel, r.dsig)
		}

		// compute weight sensitivity
		var wSens mat.Dense
		wSens.Apply(func(i, j int, s float64) float64 {
			return del.At(i, 0) * r.in.AtVec(j)
		}, l.w)

		// compute the shift of weights from the sensitivities
		var wShift mat.Dense
		wShift.Scale(-learningRate, &wSens)

		// shift weights in this layer
		var w mat.Dense
		w.Sub(l.w, &wShift)
		l.w = &w

		lastDel = &del
	}
	return nil
}

func (n *net) calculate(in mat.Vector) []*result {
	var rr []*result
	for _, l := range n.layers {
		r := l.calculate(in)
		rr = append(rr, r)
		in = r.out
	}
	return rr
}

func (l *layer) calculate(in mat.Vector) *result {
	var sum mat.Dense
	sum.Mul(l.w, in)

	var out mat.Dense
	out.Apply(applySigmoid, &sum)

	var dsig mat.Dense
	dsig.Apply(func(i, j int, s float64) float64 {
		return dsigmoid(s)
	}, &sum)

	return &result{
		in:   mat.VecDenseCopyOf(in),
		out:  out.ColView(0),
		sum:  sum.ColView(0),
		dsig: dsig.ColView(0),
	}
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

func applySigmoid(_, _ int, v float64) float64 {
	return sigmoid(v)
}

func sigmoid(val float64) float64 {
	return 1.0 / (1.0 + math.Exp(-val))
}

func dsigmoid(val float64) float64 {
	s := sigmoid(val)
	return s * (1.0 - s)
}
