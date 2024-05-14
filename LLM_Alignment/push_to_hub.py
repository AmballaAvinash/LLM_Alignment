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
os.environ["HF_TOKEN"] = "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" #write token


model_id = "./saved-models/DPO_LLAMA-7B/merged_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = (0)
tokenizer.padding_side = "right" 

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map="auto",
                                            use_cache=False)



model_id = "AvinashAmballa/LLAMA-7B-8000"
model.push_to_hub('model_id',  safe_serialization=False)
tokenizer.push_to_hub("model_id")
