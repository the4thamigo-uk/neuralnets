package main

import (
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func testPerceptron() *net {
	n := newNet(2, 2)
	n.layers[0].w = testWeights0()
	return n
}

func testHiddenLayer() *net {
	n := newNet(2, 2, 2)
	n.layers[0].w = testWeights0()
	n.layers[1].w = testWeights1()
	return n
}

func testWeights0() *mat.Dense {
	return mat.NewDense(2, 2, []float64{-0.1, 0.2, -0.3, 0.4})
}

func testWeights1() *mat.Dense {
	return mat.NewDense(2, 2, []float64{0.4, -0.3, 0.2, -0.1})
}

func testIn() mat.Vector {
	return mat.NewVecDense(2, []float64{0.25, 0.5})
}

func testExp() mat.Vector {
	return mat.NewVecDense(2, []float64{0.75, 1.0})
}

func testWeightSens(t *testing.T, n *net, in, exp mat.Vector, exps ...mat.Matrix) {
	rr := n.calculate(in)
	ll := n.learn(rr, exp)

	for i, l := range ll {
		act := l.wSens
		exp := exps[i]
		var tmp mat.Dense
		tmp.Apply(func(i, j int, _ float64) float64 {
			assert.InDelta(t, act.At(i, j), exp.At(i, j), 0.00001)
			return 0
		}, act)
	}
}

func TestLearnPerceptron(t *testing.T) {
	testWeightSens(t, testPerceptron(), testIn(), testExp(),
		mat.NewDense(2, 2, []float64{
			-0.014433368389355, -0.028866734003152,
			-0.029185256988917, -0.058370522304507,
		}))
	//testWeightSens(t, newNet(2, 2, 2), in, exp, 0, 1)
	//testWeightSens(t, newNet(2, 2, 2), in, exp, 1, 0)
	//testWeightSens(t, newNet(2, 2, 2), in, exp, 1, 1)
}

func TestCalculatePerceptron(t *testing.T) {
	n := testPerceptron()
	rr := n.calculate(testIn())
	r := rr[len(rr)-1]
	assert.InDelta(t, r.out.AtVec(0), 0.518741215878535, 0.00001)
	assert.InDelta(t, r.out.AtVec(1), 0.531209373373756, 0.00001)
}

func TestCalculateHiddenLayer(t *testing.T) {
	n := testHiddenLayer()
	rr := n.calculate(testIn())
	r := rr[len(rr)-1]
	assert.InDelta(t, r.out.AtVec(0), 0.512031095820209, 0.00001)
	assert.InDelta(t, r.out.AtVec(1), 0.512654123734249, 0.00001)
}
