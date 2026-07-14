#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# =========================================
# ResMoE Patch+Trend 动态路由 / avg 实验脚本
# 固定 seq_len=96，预测长度分别为 96/192/336/720
# 使用之前最优参数
# =========================================

model_name=ResMoE
model_variant=avg   # 可改为 router 测试动态路由
seq_len=96
label_len=48
patch_len=4
use_revin=1
enc_in=7
dec_in=7
c_out=7
e_layers=2
d_layers=1
factor=1
itr=1
batch_size=32
grid_size=5
seed=2

# 日志目录
mkdir -p logs


# --------- 预测长度 96 ---------
pred_len=96
dropout=0.5
learning_rate=0.0005
hidden_trend=64
train_epochs=20
log_file="logs/etth1_patch_trend/${model_variant}_pred96.log"

echo "🚀 ETTh1 Patch+Trend ${model_variant} pred_len=${pred_len}" > $log_file
echo "⏰ 开始时间: $(date)" >> $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id etth1_${model_variant}_pred96 \
  --model $model_name \
  --model_variant $model_variant \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --patch_len $patch_len \
  --grid_size $grid_size \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --hidden_trend $hidden_trend \
  --use_revin $use_revin \
  --itr $itr \
  --seed 2 2>&1 | tee logs/etth1_patch_trend/${model_variant}_pred96.log


# --------- 预测长度 192 ---------
pred_len=192
dropout=0.55
learning_rate=0.001
hidden_trend=128
train_epochs=25
log_file="logs/etth1_patch_trend/${model_variant}_pred192.log"

echo "🚀 ETTh1 Patch+Trend ${model_variant} pred_len=${pred_len}" > $log_file
echo "⏰ 开始时间: $(date)" >> $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id etth1_${model_variant}_pred192 \
  --model $model_name \
  --model_variant $model_variant \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --patch_len $patch_len \
  --grid_size $grid_size \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --hidden_trend $hidden_trend \
  --use_revin $use_revin \
  --itr $itr \
  --seed 2 2>&1 | tee logs/etth1_patch_trend/${model_variant}_pred192.log

'''

# --------- 预测长度 336 ---------
pred_len=336
dropout=0.65
learning_rate=0.007
hidden_trend=128
train_epochs=30
log_file="logs/etth1_patch_avg/${model_variant}_pred336.log"

echo "⏰ 开始时间: $(date)" >> $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id etth1_${model_variant}_pred336 \
  --model $model_name \
  --model_variant $model_variant \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --patch_len $patch_len \
  --grid_size $grid_size \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --hidden_trend $hidden_trend \
  --use_revin $use_revin \
  --itr $itr \
  --seed 2 2>&1 | tee logs/ETTh1_96_avg/patch_trend_avg_revin.log




# --------- 预测长度 720 ---------
pred_len=720
dropout=0.75
learning_rate=0.007
hidden_trend=128
train_epochs=25
log_file="logs/etth1_patch_trend/${model_variant}_pred720.log"

echo "🚀 ETTh1 Patch+Trend ${model_variant} pred_len=${pred_len}" > $log_file
echo "⏰ 开始时间: $(date)" >> $log_file

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh1.csv \
  --model_id etth1_${model_variant}_pred720 \
  --model $model_name \
  --model_variant $model_variant \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --patch_len $patch_len \
  --grid_size $grid_size \
  --dropout $dropout \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --batch_size $batch_size \
  --hidden_trend $hidden_trend \
  --use_revin $use_revin \
  --itr $itr \
  --seed 2 2>&1 | tee logs/ETTh1_96_avg/patch_trend_avg_revin.log
  '''