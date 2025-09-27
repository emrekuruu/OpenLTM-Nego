#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
model_name=timer
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./data/oracle \
  --model_id negotiation \
  --model $model_name \
  --data Negotiation \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 3 \
  --valid_last \
  --patience 3 \
  --covariate \
  --test_seq_len 1000 \
  --test_pred_len 1000 \
  --test_dir forecast_negotiation_timer_Negotiation_sl960_it96_ot96_lr0.0001_bt16_wd0_el3_dm1024_dff2048_nh8_cosFalse_test_0