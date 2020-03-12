#!/bin/bash
python main.py --aepoch 10 --alr 0.01 --wepoch 200 --lr 0.02 --lr_decay --batchsize 128 --outdir log/vgg_19_2 --model vgg --ratio 0.5 --dataset cifar10
