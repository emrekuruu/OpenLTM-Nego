#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=4
token_len=480
seq_len=$[$token_num*$token_len]

# training one model with a context length

python -u run.py \
  --gpu 0 \
  --task_name forecast \
  --is_training 1 \
  --root_path ./data/oracle \
  --model_id negotiation \
  --model $model_name \
  --data Negotiation \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --e_layers 4 \
  --d_model 1024 \
  --d_ff 2048 \
  --lradj type1 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --valid_last \
  --test_seq_len $seq_len \
  --test_pred_len $seq_len \


