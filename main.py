import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import os

# ===================================================================================
# 1. 配置参数 (Constants and Configuration)
# ===================================================================================

# 模型ID: 我们选择一个在中文上表现不错，且对硬件要求相对友好的模型。
# Qwen1.5-1.8B-Chat 是阿里巴巴开源的，性能强劲，且可免费商用。
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"

# 数据集路径: 指向我们刚刚创建的“教材”
DATASET_PATH = "data/puns.jsonl"

# 微调后模型的保存路径: 训练完成后，模型的“灵魂”（LoRA适配器）会保存在这里
OUTPUT_DIR = "output/puns-generator-checkpoint"


# ===================================================================================
# 2. 核心训练函数 (The Main Training Function)
# ===================================================================================

def train():
    """
    这个函数封装了整个模型的微调流程。
    """
    print(">>> 开始执行模型微调流程...")

    # --- 数据集加载 ---
    print(f"--> 步骤1: 加载数据集 '{DATASET_PATH}'")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # --- 模型和分词器加载 ---
    print(f"--> 步骤2: 加载基础模型和分词器 '{BASE_MODEL_ID}'")

    # 使用4-bit量化技术，极大降低模型对显存的需求
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 加载模型，并应用量化配置
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto" # 自动将模型分配到可用硬件（如GPU）
    )
    # 禁用模型的缓存功能，这在微调时是推荐的做法
    model.config.use_cache = False 

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    # 设置填充Token。对于Qwen模型，我们用eos_token（序列结束符）作为填充符
    tokenizer.pad_token = tokenizer.eos_token
    # 设置填充符在左边，这对于单批次生成更友好
    tokenizer.padding_side = "left"

    # --- LoRA参数配置 ---
    print("--> 步骤3: 配置LoRA (参数高效微调)")
    lora_config = LoraConfig(
        r=16,                # LoRA的秩，数值越高，可训练参数越多，但可能过拟合
        lora_alpha=32,       # LoRA的缩放因子
        lora_dropout=0.05,   # Dropout率，防止过拟合
        bias="none",
        task_type="CAUSAL_LM" # 任务类型为因果语言模型
    )

    # --- 训练参数配置 ---
    print("--> 步骤4: 配置训练参数")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,               # 指定输出目录
        per_device_train_batch_size=4,       # 每个GPU的批处理大小
        gradient_accumulation_steps=4,       # 梯度累积，用时间换空间，模拟更大的批处理大小
        learning_rate=2e-4,                  # 学习率
        max_steps=150,                       # 总训练步数，对于我们的小数据集，150步足够了
        logging_steps=10,                    # 每10步打印一次日志
        save_steps=50,                       # 每50步保存一次检查点
        fp16=True,                           # 启用半精度训练，节省显存
        push_to_hub=False,                   # 不推送到Hugging Face Hub
        report_to="none",                    # 不将结果报告给第三方平台
    )

    # --- 初始化并启动训练器 ---
    print("--> 步骤5: 初始化SFTTrainer并开始训练")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="messages", # **重要**: 告诉Trainer我们的数据在'messages'字段里
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=1024,             # 最大序列长度
        packing=True,                    # "打包"技术，将多个短序列合并成一个长序列，提升训练效率
    )

    # 开始训练
    trainer.train()

    # --- 保存最终的模型 ---
    print(f"--> 步骤6: 训练完成，保存最终模型到 '{OUTPUT_DIR}'")
    trainer.save_model(OUTPUT_DIR)
    
    print(">>> 模型微调流程圆满结束！")


# ===================================================================================
# 3. 脚本执行入口 (Script Entry Point)
# ===================================================================================
if __name__ == "__main__":
    # 当我们直接运行 `python main.py` 时，这个部分的代码会被执行
    train()