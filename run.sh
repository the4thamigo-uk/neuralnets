#!/bin/bash

go run main.go > points.tsv

gnuplot -e "set autoscale;set zrange [-1:1]; set term png;splot 'points.tsv' using 1:2:3 with points palette pointsize 1 pointtype 5" > fitted.png
gnuplot -e "set autoscale;set zrange [-1:1]; set term png;splot 'points.tsv' using 1:2:4 with points palette pointsize 1 pointtype 5" > actual.png
gnuplot -e "set autoscale;set zrange [-1:1]; set term png;splot 'points.tsv' using 1:2:5 with points palette pointsize 1 pointtype 5" > cost.png
