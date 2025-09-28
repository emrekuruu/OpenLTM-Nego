#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=timer_xl
token_num=10
token_len=96
seq_len=$[$token_num*$token_len]

training one model with a context length
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
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --lradj type1 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --valid_last \
  --covariate \
  --test_seq_len $seq_len \
  --test_pred_len $seq_len \


# testing the model on all forecast lengths
for test_pred_len in 96 960 1920
do
python -u run.py \
  --gpu 0 \
  --task_name forecast \
  --is_training 0 \
  --root_path ./data/oracle \
  --model_id negotiation \
  --model $model_name \
  --data Negotiation \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --lradj type1 \
  --use_norm \
  --cosine \
  --tmax 10 \
  --valid_last \
  --covariate \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --visualize \
  --stride 1000 \
  --test_dir forecast_negotiation_timer_xl_Negotiation_sl960_it96_ot96_lr0.0001_bt128_wd0_el1_dm512_dff2048_nh8_cosTrue_test_0
  done