#!/bin/bash

#run me with nohup ./train_rnabert.sh &> train_rnabert.out &
export KMER=6
export MODEL_PATH=../dataset/pre_trained_DNABERT/6-new-12w-0
export TRAIN_FILE=../dataset/pre_trained_DNABERT/rna_data/test_6mers.txt #train_6mers
export TEST_FILE=../dataset/pre_trained_DNABERT/rna_data/fake_6mers.txt #val_6mers
export SOURCE=../DNABERT_dependencies
export OUTPUT_PATH=../dataset/pre_trained_DNABERT/output$KMER

python train_rnabert.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --num_train_epochs 1 \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 1 \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8   \
    --save_steps 1000 \
    --save_total_limit 30 \
    --max_steps -1 \
    --logging_steps 100 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --n_process 24 