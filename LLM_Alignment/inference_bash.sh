module load miniconda/22.11.1-1 cuda/11.3.1

conda activate llm_alignment

python gpu_check.py

python LLM_Alignment/sft_inference.py

python DPO_inference.py