import os
import gc
import torch
import tqdm as notebook_tqdm
from tqdm import tqdm
import argparse
import sys
sys.path.append("..")
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from peft import prepare_model_for_int8_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, load_from_disk
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import GenerationConfig
from prompter import Prompter

class Inferencer:
    def __init__(self, input_args):
        self.input_args = input_args
        self.get_tokenizer()
        # self.get_dataset()

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.input_args.model_name_or_path)
        
        # unk. we want this to be different from the eos token
        tokenizer.pad_token_id = (0)
        
        # Allow batched inference
        tokenizer.padding_side = "right"  
        
        self.tokenizer = tokenizer
        return
    
    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.input_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.input_args.max_seq_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"]
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        
            
        tokenized_full_prompt['text'] = full_prompt
        return tokenized_full_prompt

    def get_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.bnb_config = bnb_config
        return

    def get_dataset(self, data_path, val_set_size=0):
        self.prompter = Prompter("alpaca")
        with open(data_path, 'rb') as f:
            data = json.load(f)
        data = Dataset.from_dict({
            "instruction": data["instructions"]
        })
        
        self.eval_dataset = data.map(self.generate_and_tokenize_prompt)
        

    def get_inference_model(self):
        self.get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path, 
                                                    quantization_config=self.bnb_config,
                                                    device_map='auto',
                                                    use_cache=False)
        return model
    
    

eval_datasets = [
    'I-Alpaca.json',
    'I-CoNa.json',
    'I-Controversial.json',
    'I-MaliciousInstructions.json',
    'I-PhysicalSafetySafe.json',
    'I-PhysicalSafetyUnsafe.json',
]
eval_datasets_root = "LLM_Alignment"
eval_resps_save_root = "LLM_Alignment/output"



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SFT Training Arguments")

    parser.add_argument("--model_name_or_path", type=str, default="./saved-models/DPO_LLAMA-7B/merged_model", help="Model name or path")
    parser.add_argument("--save_name", type=str, default="llama-it", help="Training Data Path")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum Sequence length")

    input_args = parser.parse_args()

    inferencer = Inferencer(input_args)
    model = inferencer.get_inference_model()
    print("### model loaded")


    for d in tqdm(eval_datasets):
        eval_data = f"{eval_datasets_root}/{d}"
        print(f"Eval data: {eval_data}")
        inferencer.get_dataset(eval_data)
        responses = {
            "instruction": [],
            "response": []
        }
        n_rows = len(inferencer.eval_dataset)
        for i in tqdm(range(n_rows)):
            prompt = inferencer.eval_dataset[i]['text']#prompter.generate_prompt(instruction, input)
            inputs = inferencer.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to('cuda')
            generation_config = GenerationConfig(
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=4,
                do_sample=True,
                # **kwargs,
            )

            generate_params = {
                "input_ids": input_ids,
                "generation_config": generation_config,
                "return_dict_in_generate": True,
                "output_scores": True,
                "max_new_tokens": 128,
            }

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128,
                )
            s = generation_output.sequences[0]
            output = inferencer.tokenizer.decode(s[input_ids.shape[-1]:], skip_special_tokens=True)
            responses["instruction"].append(inferencer.eval_dataset[i]["instruction"])
            responses["response"].append(output)

        save_path = f"{eval_resps_save_root}/{input_args.save_name}/{d}"
        with open(save_path, 'w') as f:
            json.dump(responses, f)