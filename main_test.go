package main

import (
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func testPerceptron() *net {
	n := newNet(2, 2)
	n.layers[0].w = testWeights0()
	n.layers[0].b = testBias0()
	return n
}

func testHiddenLayer() *net {
	n := newNet(2, 2, 2)
	n.layers[0].w = testWeights0()
	n.layers[1].w = testWeights1()
	n.layers[0].b = testBias0()
	n.layers[1].b = testBias1()
	return n
}

func testWeights0() mat.Matrix {
	return mat.NewDense(2, 2, []float64{-0.1, 0.2, -0.3, 0.4})
}

func testBias0() mat.Vector {
	return mat.NewVecDense(2, []float64{0.1, -0.2})
}

func testWeights1() mat.Matrix {
	return mat.NewDense(2, 2, []float64{0.4, -0.3, 0.2, -0.1})
}

func testBias1() mat.Vector {
	return mat.NewVecDense(2, []float64{0.3, -0.4})
}

func testIn() mat.Vector {
	return mat.NewVecDense(2, []float64{0.25, 0.5})
}

func testExp() mat.Vector {
	return mat.NewVecDense(2, []float64{0.75, 1.0})
}

func testWeightSens(t *testing.T, n *net, in, exp mat.Vector, wSens []mat.Matrix, bSens []mat.Vector) {
	rr := n.calculate(in)
	ll := n.learn(rr, exp)

	for i, l := range ll {
		act := l.wSens
		exp := wSens[i]
		var tmp mat.Dense
		tmp.Apply(func(i, j int, _ float64) float64 {
			assert.InDelta(t, act.At(i, j), exp.At(i, j), 0.00001)
			return 0
		}, act)
	}

	for i, l := range ll {
		act := l.bSens
		exp := bSens[i]
		var tmp mat.Dense
		tmp.Apply(func(i, j int, _ float64) float64 {
			assert.InDelta(t, act.At(i, j), exp.At(i, j), 0.00001)
			return 0
		}, act)
	}

	n.adjust(ll)

}

func TestLearnPerceptron(t *testing.T) {
	testWeightSens(t, testPerceptron(), testIn(), testExp(),
		[]mat.Matrix{
			mat.NewDense(2, 2, []float64{
				-0.012799336590597, -0.025598670405636,
				-0.032375774283722, -0.064751548567443,
			}),
		},
		[]mat.Vector{
			mat.NewVecDense(2, []float64{
				-0.051197343864384,
				-0.129503101575779,
			}),
		},
	)
}

func TestLearnHiddenLayer(t *testing.T) {
	testWeightSens(t, testHiddenLayer(), testIn(), testExp(),
		[]mat.Matrix{
			mat.NewDense(2, 2, []float64{
				-0.002705501656042, -0.005411003312084,
				0.001599058940815, 0.003198117881631,
			}),
			mat.NewDense(2, 2, []float64{
				-0.020716925397402, -0.018339758878216,
				-0.077134561793368, -0.068283744769637,
			}),
		},
		[]mat.Vector{
			mat.NewVecDense(2, []float64{
				-0.010822007456834,
				0.006396235763262,
			}),
			mat.NewVecDense(2, []float64{
				-0.038107892397221,
				-0.141885710958078,
			}),
		},
	)
}

func TestCalculatePerceptron(t *testing.T) {
	n := testPerceptron()
	rr := n.calculate(testIn())
	r := rr[len(rr)-1]
	assert.InDelta(t, r.out.AtVec(0), 0.543638687237079, 0.00001)
	assert.InDelta(t, r.out.AtVec(1), 0.481258784121465, 0.00001)
}

func TestCalculateHiddenLayer(t *testing.T) {
	n := testHiddenLayer()
	rr := n.calculate(testIn())
	r := rr[len(rr)-1]
	assert.InDelta(t, r.out.AtVec(0), 0.592202483123707, 0.00001)
	assert.InDelta(t, r.out.AtVec(1), 0.415955683223968, 0.00001)
}
