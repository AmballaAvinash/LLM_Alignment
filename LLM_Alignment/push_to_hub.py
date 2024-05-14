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

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import notebook_login


# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models


# Login to HF Hub
notebook_login()


model_id = "./saved-models/DPO_LLAMA-7B/merged_model"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)


model.push_to_hub('LLAMA-7B-8000')
tokenizer.push_to_hub("LLAMA-7B-8000")
