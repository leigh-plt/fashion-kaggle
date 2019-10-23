#!/bin/bash

TPU_IP_ADDRESS=10.128.17.2

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=0

python src/train.py \
     --epochs 200 \
     --model_name fcn_resnet50 \
     --log_step 0 \
     --batch_size 16 \
     --num_cores 8 \
     --num_worker 4 \
     --lr 1e-3 \
     --weight_decay 1e-4 \
     --slr_divisor 2 \
     --slr_divide_n_epochs 20 \
     --num_warmup_epochs 0.5 \
     --min_lr 1e-5 \
     --data_path 'data/train' \
     --json_file 'data/train.json' \
     --log_file 'report/train.fcn_resnet50.log' \
     --save_file 'checkpoint/fcn_resnet50.pth'

# python src/train.py \
#      --epochs 50 \
#      --model_name deeplabv3_resnet50 \
#      --log_step 0 \
#      --batch_size 16 \
#      --num_cores 8 \
#      --num_worker 4 \
#      --lr 1e-3 \
#      --weight_decay 1e-4 \
#      --slr_divisor 2 \
#      --slr_divide_n_epochs 5 \
#      --num_warmup_epochs 0.5 \
#      --min_lr 1e-4 \
#      --data_path 'data/train' \
#      --json_file 'data/train.json' \
#      --log_file 'report/train.deeplabv3_resnet50.log' \
#      --save_file 'checkpoint/deeplabv3_resnet50.pth'