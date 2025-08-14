# debug_inference.py
# 这是一个专门用于测试和调试推理流程的脚本。

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import os

# --- 使用和main.py完全相同的常量 ---
BASE_MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
OUTPUT_DIR = "output/puns-generator-checkpoint"

print("--- 开始执行推理调试脚本 ---")

try:
    # --- 1. 加载基础模型和分词器 ---
    print(f"--> 步骤1: 正在加载基础模型 '{BASE_MODEL_ID}'...")
    # 我们在CPU上加载以进行测试
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print("--> 基础模型和分词器加载成功！")

    # --- 2. 加载LoRA适配器 ---
    print(f"--> 步骤2: 正在加载LoRA适配器 '{OUTPUT_DIR}'...")
    # PeftModel 会将LoRA权重合并到基础模型中
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    print("--> LoRA适配器加载成功！模型已准备就绪。")

    # --- 3. 准备输入 ---
    print("--> 步骤3: 正在准备测试输入...")
    prompt = "讲个关于程序员的笑话"
    messages = [{"role": "user", "content": prompt}]
    
    # 使用分词器的聊天模板来格式化输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    print(f"--> 准备好的输入文本: {text}")

    # --- 4. 执行生成 (最关键的测试步骤) ---
    print("--> 步骤4: 正在调用 model.generate() ...")
    
    # 使用我们之前确定的稳定生成配置
    generation_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated_ids = model.generate(
        model_inputs.input_ids,
        generation_config=generation_config
    )
    print("--> model.generate() 调用成功！")

    # --- 5. 解码输出 ---
    print("--> 步骤5: 正在解码输出...")
    decoded_output = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print("\n===================================")
    print("      🎉 调试成功！🎉")
    print("===================================")
    print(f"模型生成的回答是: {decoded_output}")

except Exception as e:
    print("\n===================================")
    print("      🔥 调试失败！🔥")
    print("===================================")
    print(f"在执行过程中遇到了错误: {e}")
    import traceback
    traceback.print_exc()