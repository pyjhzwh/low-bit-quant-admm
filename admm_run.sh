#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=all_cnn_net_log/cifar10

mkdir -p ${DIR}
mkdir -p saved_models

echo "" > ${DIR}/admm.layerwise.log

for conv0 in 1 2
do
    for conv1 in 1 2
    do
        for conv2 in 1 2
        do
            for conv3 in 1 2
            do
                for conv4 in 1 2
                do
                    for conv5 in 1 2
                    do
                        for conv6 in 1 2
                        do
                            for conv7 in 1 2
                            do
                                for conv8 in 1 2
                                do
                                    echo "" >> ${DIR}/admm.layerwise.log
                                    echo ${conv0},${conv1},${conv2},${conv3},${conv4},${conv5},${conv6},${conv7},${conv8} >> ${DIR}/admm.layerwise.log
                                    CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 450 --wd 1e-3 --admm --admm-iter 10 --pretrained saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar --bits ${conv0} ${conv1} ${conv2} ${conv3} ${conv4} ${conv5} ${conv6} ${conv7} ${conv8}  &> ${DIR}/admm.pretrained.log 2>&1
                                    cat ${DIR}/admm.pretrained.log | tail -n12 >> ${DIR}/admm.layerwise.log
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
    
#CUDA_VISIBLE_DEVICES=0,1 python3 main.py --arch all_cnn_c --dataset cifar10 --lr 1e-2 --epochs 450 --wd 1e-3 --admm --admm-iter 10 --pretrained saved_models/best.all_cnn_c.32.32.ckp_origin.pth.tar --bits 2 1 2 2 2 2 2 2 2  &> ${DIR}/admm.pretrained.log 2>&1

<<COMMAND1
for SIZE in 8 12 16 20 
do
    CUDA_VISIBLE_DEVICES=0,3 python3 main.py --ds ${SIZE} &> ${DIR}/small.${SIZE}.log
done
COMMAND1
