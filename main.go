package main

import (
	"fmt"
	"github.com/davecgh/go-spew/spew"
	"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"os"
	"time"
)

const (
	defaultWeight = 0.0
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

	learned struct {
		wSens  mat.Matrix
		wShift mat.Matrix
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
	n := newNet(2, 10, 10, 1)

	f := func(x, y float64) float64 {
		return y * math.Sin(math.Pi*x)
	}

	const N = 10000000

	rand.Seed(time.Now().UnixNano())

	for i := 0; i < N; i++ {
		x := 2.0*rand.Float64() - 1.0
		y := 2.0*rand.Float64() - 1.0
		z := f(x, y)
		in := mat.NewVecDense(2, []float64{x, y})
		exp := mat.NewVecDense(1, []float64{z})

		rr := n.calculate(in)
		n.learn(rr, exp)

		if i%100000 == 0 {
			_, cost := cost(rr[len(rr)-1].out, exp)
			spew.Printf("train: epoch=%d, cost=%v\n", i*100.0/N, sum(cost))
		}
	}

	spew.Println("x\ty\tz\tact\tcost")

	for x := -1.0; x <= 1.0; x += 0.1 {
		for y := -1.0; y <= 1.0; y += 0.1 {
			z := f(x, y)
			in := mat.NewVecDense(2, []float64{x, y})
			exp := mat.NewVecDense(1, []float64{z})

			rr := n.calculate(in)
			_, cost := cost(rr[len(rr)-1].out, exp)
			spew.Printf("%v\t%v\t%v\t%v\t%v\n", x, y, rr[len(rr)-1].out.AtVec(0), z, sum(cost))
		}
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

func (n *net) learn(rr []*result, exp mat.Vector) []*learned {
	var lastDel mat.Matrix

	out := make([]*learned, len(rr))

	for i := len(rr) - 1; i >= 0; i-- {
		r := rr[i]
		l := n.layers[i]

		var del mat.Dense
		if i == len(rr)-1 /* output layer */ {

			diff, _ := cost(r.out, exp)
			del.MulElem(diff, r.dsig)

		} else /* hidden layers */ {
			var wdel mat.Dense
			wdel.Mul(n.layers[i+1].w.T(), lastDel)
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
		w.Add(l.w, &wShift)
		l.w = &w

		out[i] = &learned{
			wSens:  &wSens,
			wShift: &wShift,
		}

		lastDel = &del
	}
	return out
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

func cost(out, exp mat.Vector) (mat.Vector, mat.Vector) {
	var diff mat.Dense
	diff.Sub(out, exp)

	var cost mat.Dense
	N := float64(out.Len())
	cost.Apply(func(_, _ int, d float64) float64 {
		return 1.0 / N * d * d
	}, &diff)

	return diff.ColView(0), cost.ColView(0)
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

func sum(v mat.Vector) float64 {
	sum := 0.0
	for i := 0; i < v.Len(); i++ {
		sum += v.AtVec(i)
	}
	return sum
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
