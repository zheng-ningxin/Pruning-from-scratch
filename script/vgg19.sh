#!/bin/bash
python main.py --aepoch 10 --alr 0.01 --wepoch 500 --lr 0.02 --lr_decay --batchsize 128 --outdir log/vgg_19_4 --model vgg --ratio 0.5 --dataset cifar10
