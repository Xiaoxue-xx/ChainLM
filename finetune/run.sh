set -x

NNODES=$1
NODE_RANK=$2
MASTER_ADDR=$3

export WANDB_MODE=offline
export OMP_NUM_THREADS=24

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=11452 \
    train_multi.py \
    --model_name_or_path [foundation_model_path] \
    --data_path ./data/train_data.json \
    --bf16 True \
    --output_dir [output_dir] \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --deepspeed ds_z3_bf16.json \
    --gradient_checkpointing True \
    --tf32 True