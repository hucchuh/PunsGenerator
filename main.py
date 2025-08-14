# ===================================================================================
# PunsGenerator - main.py (Final Training Script)
# Version: 1.2 - Switched to BF16 for training stability
# Date: 2025-08-14
# ===================================================================================

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, GenerationConfig
from peft import LoraConfig, PeftModel
import gradio as gr
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

# if __name__ == "__main__":
#     train()

# ===================================================================================
# Gradio App Function (Final Stability Fixes)
# ===================================================================================
def run_gradio_app():
    print(">>> Launching Gradio application...")

    print(f"--> Step 1: Loading base model in FP32 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    print(f"--> Step 2: Loading LoRA adapter from '{OUTPUT_DIR}'")
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    
    print("--> Step 2.5: Setting model to evaluation mode.")
    model.eval()
    print("--> Model is ready for inference!")

    def generate_pun(prompt, history):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

        print("--> Generating response with stable configuration...")
        with torch.no_grad():
            # **关键修正 1**: 显式创建一个GenerationConfig对象
            generation_config = GenerationConfig(
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8,    # **修正 2**: 稍微提高温度，让概率分布更平滑，减少极端值的出现
                top_k=50,           # **修正 3**: 使用top_k代替top_p。Top_k更直接地限制了词汇选择范围，有时在CPU上更稳定
                top_p=0.9,          
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            generated_ids = model.generate(
                **model_inputs, # 使用 ** 展开字典作为参数
                generation_config=generation_config,
                use_cache=False # **修正 4**: 明确禁用KV缓存，这有时是CPU推理不稳定的根源
            )
        
        decoded_output = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        print(f"--> Model response: {decoded_output}")
        return decoded_output

    print("--> Step 3: Creating and launching Gradio interface")
    iface = gr.ChatInterface(
        fn=generate_pun,
        title="谐音梗大师 (PunsGenerator) 🤖 - Qwen1.5版",
        description="输入任何主题，我会努力为你生成一个有趣的谐音梗或冷笑话！",
        examples=[["讲个关于程序员的笑话"], ["有什么关于食物的梗吗？"]]
    )
    iface.launch(share=True)

# ===================================================================================
# Script Entry Point
# ===================================================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        train()
    else:
        run_gradio_app()