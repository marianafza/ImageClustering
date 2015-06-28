#!/bin/bash

for numpatch in 16 25 36 64 128
do
    for size in 36 64 128 256
    do
    
    python imageClustering.py --met BOC -p /Users/mariana/tese/datasets/natural -n $numpatch -g SAMPLEP:0.1 -c MINIBATCH -t $size -f NONE -h NONE -a KMEANS -m euclidean --nclust 8 -r 10
    
    done
done