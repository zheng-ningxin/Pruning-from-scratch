#!/bin/bash
python main.py --aepoch 10 --alr 0.01 --wepoch 160 --lr 0.1 --lr_decay --batchsize 64 --outdir log/vgg19_5 --model vgg --ratio 0.5 --dataset cifar10
