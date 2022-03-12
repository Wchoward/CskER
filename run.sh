#!/bin/bash
# export DATASET=ISEAR
# export MODEL_NAME=bert_base
# export PRE_TRAIN_PTH=/home/ssm/project/wch/pretrained_models/bert_base_uncased_eng
export CUDA_VISIBLE_DEVICES=1

# # ISEAR dataset
# python run.py \
#     --log_dir log \
#     --dataset ISEAR \
#     --model_name bert_attention \
#     --pre_train_path  /home/ssm/project/wch/pretrained_models/bert_base_uncased_eng \
#     --output_dir output \
#     --do_train \
#     --do_eval \
#     --max_seq_length 140 \
#     --do_lower_case \
#     --num_train_epochs 20 \
#     --train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
# 	--learning_rate 1e-5 \
# 	--weight_decay 0.02 \
# 	--warmup_rate 0.3 \

# # TEC dataset
# python run.py \
#     --log_dir log \
#     --dataset TEC \
#     --model_name bert_attention \
#     --pre_train_path  /home/ssm/project/wch/pretrained_models/bert_base_uncased_eng \
#     --output_dir output \
#     --do_train \
#     --do_eval \
#     --max_seq_length 140 \
#     --do_lower_case \
#     --num_train_epochs 20 \
#     --train_batch_size 2 \
#     --gradient_accumulation_steps 16 \
# 	--learning_rate 1e-5 \
# 	--weight_decay 0.02 \
# 	--warmup_rate 0.3 \


# # ISEAR dataset
# python run_csk.py \
#     --log_dir log \
#     --dataset ISEAR \
#     --model_name bert_csk_attention \
#     --pre_train_path  /home/ssm/project/wch/pretrained_models/bert_base_uncased_eng \
#     --output_dir output \
#     --do_train \
#     --do_eval \
#     --max_seq_length 256 \
#     --do_lower_case \
#     --num_train_epochs 20 \
#     --train_batch_size 8 \
#     --eval_batch_size 8 \
#     --gradient_accumulation_steps 8 \
# 	--learning_rate 1e-5 \
# 	--weight_decay 0.02 \
# 	--warmup_rate 0.3 \

# TEC dataset
python run.py \
    --log_dir log \
    --dataset TEC \
    --model_name bert_lstm_attention \
    --pre_train_path  /home/ssm/project/wch/pretrained_models/bert_base_uncased_eng \
    --output_dir output \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --do_lower_case \
    --num_train_epochs 20 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
	--learning_rate 5e-5 \
	--weight_decay 0.02 \
	--warmup_rate 0.3 

python run_csk.py \
    --log_dir log \
    --dataset TEC \
    --model_name bert_csk_lstm_attention \
    --pre_train_path  /home/ssm/project/wch/pretrained_models/bert_base_uncased_eng \
    --output_dir output \
    --do_train \
    --do_eval \
    --max_seq_length 256 \
    --do_lower_case \
    --num_train_epochs 20 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
	--learning_rate 5e-5 \
	--weight_decay 0.02 \
	--warmup_rate 0.3 