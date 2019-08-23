#! /bin/bash
export LD_LIBRARY_PATH=/home/panyj/.local/lib/:$LD_LIBRARY_PATH

DIR=all_cnn_net_log/imagenet

mkdir -p ${DIR}
mkdir -p saved_models

#CUDA_VISIBLE_DEVICES=0,1 python3 main.py --ds 224 --crop 224 --arch 'resnet18' --dataset 'imagenet' -e #&> ${DIR}/big.log


#for SIZE in 56 112 168 196 
#do
#    CUDA_VISIBLE_DEVICES=0,1 python3 main.py --ds 224 --crop ${SIZE} --arch 'resnet18' --dataset 'imagenet' -e  &> ${DIR}/small.crop${SIZE}.val.log
#done

<<COMMENT1
LOGPATH=${DIR}/test_ds_crop.log
echo "" >> ${LOGPATH}

for DS in 84
do
    for CROP in 84 112 168 196 224
    do
        #if [ "${DS}" -le "${CROP}" ]; 
        #then
            echo -e "\n ds=${DS}, crop=${CROP}" >> ${LOGPATH}
            CUDA_VISIBLE_DEVICES=0,1 python3 main.py --ds ${DS} --crop ${CROP} --arch 'resnet18' --dataset 'imagenet' -e  | tail -n1 >> ${LOGPATH}
        #fi

    done
done
COMMENT1
#<<COMMENT2
DS=224
CROP=224
CUDA_VISIBLE_DEVICES=2,3 python3 main.py --ds ${DS} --crop ${CROP} --arch 'all_cnn_net' --dataset 'imagenet' --epochs 100 --lr 1e-3 --lr_epochs 50 -e &> ${DIR}/small.crop${CROP}.ds${DS}.log
#COMMENT2

