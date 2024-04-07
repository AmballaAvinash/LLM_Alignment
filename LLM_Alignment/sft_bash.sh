module load miniconda/22.11.1-1 cuda/11.3.1

conda activate llm_alignment

python gpu_check.py

python LLM_Alignment/sft_trainer.py \
      --model_name_or_path google/gemma-2b \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
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



python LLM_Alignment/DPO_trainer.py \
      --model_name_or_path google/gemma-2b \
      --per_device_train_batch_size 8 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 1 \
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
