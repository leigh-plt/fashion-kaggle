#!/bin/bash

TPU_IP_ADDRESS=10.128.17.2

export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
export XLA_USE_BF16=0

python src/inference.py \
     --model_name fcn_resnet50 \
     --data_path 'data/test' \
     --csv_file 'submission.csv' \
     --pht_file 'checkpoint/fcn_resnet50.pth'