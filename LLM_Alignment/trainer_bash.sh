#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=50G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 4:00:00  # Job time limit
#SBATCH --constraint=vram48
#SBATCH -o slurm-%j.out  # %j = job ID
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"


module load miniconda/22.11.1-1 cuda/11.3.1

conda activate NLP685

python LLM_Alignment/gpu_check.py



'python LLM_Alignment/sft_trainer.py \
      --model_name_or_path google/gemma-2b \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --gradient_accumulation_steps 1 \
      --learning_rate 5e-5 \
      --report_to wandb \
      --run_name SFT_training \
      --max_seq_length 1024 \
      --num_train_epochs 2 \
      --evaluation_strategy steps \
      --eval_steps 30 \
      --logging_strategy steps \
      --log_steps 30 \
      --logging_first_step \
      --save_strategy epoch \
      --save_steps 1 \
      --lora_rank 8 \
      --lora_alpha 32 \
      --lora_dropout 0.1 \
      --output_dir ./saved-models/SFT
'



python LLM_Alignment/DPO_trainer.py \
      --model_name_or_path google/gemma-2b \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 4 \
      --learning_rate 5e-5 \
      --report_to wandb \
      --run_name DPO_Training \
      --max_seq_length 1024 \
      --num_train_epochs 2 \
      --evaluation_strategy steps \
      --eval_steps 30 \
      --logging_strategy steps \
      --log_steps 30 \
      --logging_first_step \
      --save_strategy epoch \
      --save_steps 1 \
      --lora_rank 8 \
      --lora_alpha 32 \
      --lora_dropout 0.1 \
      --output_dir ./saved-models/DPO
