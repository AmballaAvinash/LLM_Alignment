#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 00:17:07 2024

@author: avinashamballa
"""

import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse

import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login


# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models


model_id = "./saved-models/DPO_LLAMA-7B/merged_model"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             quantization_config=bnb_config, 
                                             device_map={"": 0},
                                            use_cache=False)




# Login to HF Hub
notebook_login()


model.push_to_hub('LLAMA-7B-8000')
tokenizer.push_to_hub("LLAMA-7B-8000")
