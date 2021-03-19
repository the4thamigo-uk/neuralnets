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
	learningRate = 0.1
)

type (
	net struct {
		dim    int
		layers []*layer
	}

	layer struct {
		w          mat.Matrix
		b          mat.Vector
		activation func(float64) float64
	}

	result struct {
		in   mat.Vector
		out  mat.Vector
		dsig mat.Vector
	}

	learned struct {
		w      mat.Matrix
		b      mat.Vector
		wSens  mat.Matrix
		bSens  mat.Matrix
		wShift mat.Matrix
		del    mat.Vector
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
	rand.Seed(time.Now().UnixNano())

	n := newNet(2, 20, 20, 1)
	n.init(randWeight)

	f := func(x, y float64) float64 {
		return (1.0 + math.Cos(2*math.Pi*math.Sqrt(x*x+y*y))) / 2.0
	}

	const (
		epochs = 1000000
		update = epochs / 100
	)

	for i := 0; i < epochs; i++ {
		x := rand.Float64()
		y := rand.Float64()
		z := f(x, y)

		in := mat.NewVecDense(2, []float64{x, y})
		exp := mat.NewVecDense(1, []float64{z})

		rr := n.calculate(in)
		ll := n.learn(rr, exp)
		n.adjust(ll)

		//if i%update == 0 {
		//	_, cost := cost(rr[len(rr)-1].out, exp)
		//	spew.Printf("train %v%%: cost=%v\r", i*100.0/epochs, sum(cost))
		//}
	}

	spew.Println("x\ty\tz\tact\tdiff\tcost")

	for x := 0.0; x <= 1.0; x += 0.05 {
		for y := 0.0; y <= 1.0; y += 0.05 {
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
		w := mat.NewDense(r, c, makeSlice(r*c, 0))
		b := mat.NewVecDense(r, makeSlice(r, 0))
		lrs = append(lrs, &layer{w: w, b: b})
	}
	return &net{
		layers: lrs,
	}
}

func (n *net) init(weight func() float64) {
	for _, l := range n.layers {
		l.init(weight)
	}
}

func (n *net) learn(rr []*result, exp mat.Vector) []*learned {
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
			wdel.Mul(n.layers[i+1].w.T(), out[i+1].del)
			del.MulElem(&wdel, r.dsig)
		}

		// compute weight sensitivity
		var wSens mat.Dense
		wSens.Apply(func(i, j int, s float64) float64 {
			return del.At(i, 0) * r.in.AtVec(j)
		}, l.w)

		var bSens mat.Dense
		bSens.Apply(func(i, j int, s float64) float64 {
			return del.At(i, 0)
		}, l.b)

		// compute the shift of weights from the sensitivities
		var wShift mat.Dense
		wShift.Scale(-learningRate, &wSens)

		var bShift mat.Dense
		bShift.Scale(-learningRate, &bSens)

		// shift weights in this layer
		var w mat.Dense
		w.Add(l.w, &wShift)

		var b mat.Dense
		b.Add(l.b, &bShift)

		out[i] = &learned{
			wSens:  &wSens,
			bSens:  &bSens,
			wShift: &wShift,
			w:      &w,
			b:      b.ColView(0),
			del:    del.ColView(0),
		}
	}
	return out
}

func (n *net) adjust(ll []*learned) {
	for i, lyr := range n.layers {
		lrn := ll[i]
		lyr.w = lrn.w
		lyr.b = lrn.b
	}
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
	var mul mat.Dense
	mul.Mul(l.w, in)

	var sum mat.Dense
	sum.Add(&mul, l.b)

	var act mat.Dense
	act.Apply(applySigmoid, &sum)

	var dsig mat.Dense
	dsig.Apply(func(_, _ int, s float64) float64 {
		return dsigmoid(s)
	}, &sum)

	return &result{
		in:   mat.VecDenseCopyOf(in),
		out:  act.ColView(0),
		dsig: dsig.ColView(0),
	}
}

func (l *layer) init(weight func() float64) {
	var w mat.Dense
	w.Apply(func(_, _ int, _ float64) float64 {
		return weight()
	}, l.w)
	l.w = &w

	var b mat.Dense
	b.Apply(func(_, _ int, _ float64) float64 {
		return weight()
	}, l.b)
	l.b = b.ColView(0)
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

func randWeight() float64 {
	return rand.Float64() / 2.0
}
