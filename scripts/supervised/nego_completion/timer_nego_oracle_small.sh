#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Activate virtual environment
source .venv/bin/activate

# Negotiation Completion Training with Timer (Smaller Model)
# This script trains a smaller Timer model for faster experimentation
# Input: Variable-length negotiation history (min 1000 steps)
# Output: Complete remaining negotiation until session end

model_name=timer
data_name=NegoCompletionOracle
min_input_len=1000

# Smaller model configuration for faster training/testing
d_model=256
d_ff=1024
e_layers=4
n_heads=4

# Training configuration
batch_size=32
learning_rate=0.0002
train_epochs=20
patience=5
num_workers=4

# Sequence configuration
seq_len=1440    # Smaller max context for faster training
input_token_len=96
output_token_len=96

echo "Starting Timer (small) training for negotiation completion..."
echo "Dataset: $data_name"
echo "Model: $model_name"
echo "Min input length: $min_input_len"
echo "Batch size: $batch_size"
echo "Learning rate: $learning_rate"
echo "Epochs: $train_epochs"

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./data/openltm/oracle/test/ \
  --model_id nego_completion_oracle_small \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --input_token_len $input_token_len \
  --output_token_len $output_token_len \
  --test_seq_len $seq_len \
  --test_pred_len $output_token_len \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --n_heads $n_heads \
  --num_workers $num_workers \
  --cosine \
  --tmax $train_epochs \
  --valid_last \
  --patience $patience \
  --weight_decay 0.01 \
  --des "nego_completion_oracle_timer_small"

echo "Training completed!"