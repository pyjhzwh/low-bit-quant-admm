#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=all_cnn_net_log/cifar10

mkdir -p ${DIR}
mkdir -p saved_models

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 10 --wd 1e-3 --admm --admm-iter 1 --pretrained saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar &> ${DIR}/admm.pretrained.log 2>&1

<<COMMAND1
for SIZE in 8 12 16 20 
do
    CUDA_VISIBLE_DEVICES=0,3 python3 main.py --ds ${SIZE} &> ${DIR}/small.${SIZE}.log
done
COMMAND1
