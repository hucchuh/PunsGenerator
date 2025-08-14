import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os

# ... (前面的常量定义部分保持不变) ...
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
DATASET_PATH = "data/puns.jsonl"
OUTPUT_DIR = "output/puns-generator-checkpoint"

def train():
    print(">>> 开始执行模型微调流程 (无量化模式)...")
    print(f"--> 步骤1: 加载数据集 '{DATASET_PATH}'")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print(f"--> 步骤2: 加载基础模型和分词器 '{BASE_MODEL_ID}'")
    # **核心改动在这里：我们不再使用BitsAndBytesConfig**
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16, # 直接使用16-bit半精度加载
        device_map="auto"
    )
    model.config.use_cache = False 

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ... (LoRA配置和训练参数部分保持不变) ...
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    )
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=2e-4,
        max_steps=150, logging_steps=10, save_steps=50,
        fp16=True, push_to_hub=False, report_to="none",
    )

    print("--> 步骤5: 初始化SFTTrainer并开始训练")
    trainer = SFTTrainer(
        model=model, train_dataset=dataset, peft_config=lora_config,
        dataset_text_field="messages", args=training_args,
        tokenizer=tokenizer, max_seq_length=1024, packing=True,
    )

    trainer.train()
    print(f"--> 步骤6: 训练完成，保存最终模型到 '{OUTPUT_DIR}'")
    trainer.save_model(OUTPUT_DIR)
    print(">>> 模型微调流程圆满结束！")

if __name__ == "__main__":
    train()