SFT LLM using PEFT

Create environment (virtualenv or conda) -  Python 3.9:  conda create -n NLP685 python=3.9 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install scikit-learn
pip install pandas
pip install tensorboard
pip install trl
pip install peft bitsandbytes



DPO references:
1. https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
2. https://huggingface.co/blog/dpo-trl
