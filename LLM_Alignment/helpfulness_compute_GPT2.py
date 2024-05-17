from datasets import load_dataset
import json
import pandas as pd
import sys
from prompter import Prompter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse

import os
import json
import os.path as osp
from typing import Union

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import copy


import os
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models



import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-helpful-reward_model')
reward_model = AutoModelForSequenceClassification.from_pretrained(
                'Ray2333/gpt2-large-helpful-reward_model',
                num_labels=1, torch_dtype=torch.bfloat16,
                device_map=0,
                )
q, a = "\n\nHuman: I just came out of from jail, any suggestion of my future? \n\nAssistant:", "Sorry, I don't understand."
inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
with torch.no_grad():
    reward = reward_model(**(inputs.to(0))).logits[0].cpu().detach().item()
print(reward)


datafile = "LLM_Alignment/output/DPO_LLAMA-7B_alpaca_test.json"

with open(datafile, 'r') as f:
    data = json.load(f)
    
    

def get_rew_scores(instructions, outputs):
    all_outputs = []
    for i in tqdm(range(len(instructions))):
        q, a = f"\n\nHuman: {instructions[i]} \n\nAssistant:", outputs[i]
        inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
        with torch.no_grad():
            reward = reward_model(**(inputs.to(0))).logits[0].cpu().detach().item()
        all_outputs.append(reward)
    return all_outputs




all_outputs = []
for i in tqdm(range(len(data['instructions']))):
    q, a = f"\n\nHuman: {data['instructions'][i]} \n\nAssistant:", data['outputs'][i]
    inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
    with torch.no_grad():
        reward = reward_model(**(inputs.to(0))).logits[0].cpu().detach().item()
    all_outputs.append(reward)
    
    
    
test = all_outputs.copy()
test.sort()
print(test[475])
print("\n")
all_ops = get_rew_scores(data['instructions'], data['model_responses'])


print(sum(all_ops)/len(all_ops))
print("\n")


test_all_ops = all_ops.copy()

test_all_ops.sort()
print(test_all_ops[475])