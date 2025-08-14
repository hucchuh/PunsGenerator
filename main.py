# ===================================================================================
# PunsGenerator - main.py (Final Training Script)
# Version: 1.2 - Switched to BF16 for training stability
# Date: 2025-08-14
# ===================================================================================

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- (Constants and Configuration) ---
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DATASET_PATH = "data/puns.jsonl"
OUTPUT_DIR = "output/puns-generator-checkpoint"

def train():
    """
    This function encapsulates the entire model fine-tuning workflow.
    """
    print(">>> Starting model fine-tuning (Stable BF16 version)...")
    
    # --- Step 1: Load Dataset ---
    print(f"--> Step 1: Loading dataset '{DATASET_PATH}'")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # --- Step 2: Load Base Model and Tokenizer ---
    print(f"--> Step 2: Loading base model and tokenizer '{BASE_MODEL_ID}'")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- New Step: Manually Format Dataset ---
    print("--> New Step: Manually formatting dataset")
    def format_dataset(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    formatted_dataset = dataset.map(format_dataset)

    # --- Step 3: Configure LoRA ---
    print("--> Step 3: Configuring LoRA for PEFT")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # --- Step 4: Configure Training Arguments ---
    print("--> Step 4: Configuring Training Arguments")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=150,
        logging_steps=10,
        save_steps=50,
        bf16=True, # **CRITICAL FIX**: Use BF16 for stability on modern GPUs like T4.
        push_to_hub=False,
        report_to="none",
    )

    # --- Step 5: Initialize and Start SFTTrainer ---
    print("--> Step 5: Initializing SFTTrainer and starting training")
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,
        packing=True,
    )

    trainer.train()
    
    # --- Step 6: Save Final Model ---
    print(f"--> Step 6: Training complete. Saving final model to '{OUTPUT_DIR}'")
    trainer.save_model(OUTPUT_DIR)
    
    print(">>> Model fine-tuning workflow has finished successfully!")

if __name__ == "__main__":
    train()