import json
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import os

# 配置
MODEL_PATH = "/users/cyx/Meta-Llama-3-8B-Instruct" 
DATASET_NAME = "narrativeqa" # 或 gov_report
TARGET_LENGTH = 3000 # 目标 Token 长度

def get_prompt():
    print(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # 模拟 pred.py 的配置读取
    dataset2prompt = json.load(open("../evaluation/LongBench/config/dataset2prompt.json", "r"))
    prompt_format = dataset2prompt[DATASET_NAME]
    
    print(f"Loading dataset {DATASET_NAME} from local file...")
    # 本地加载：直接读取 jsonl 文件
    data_path = f"/users/cyx/LongBench/data/{DATASET_NAME}.jsonl"
    
    with open(data_path, "r", encoding="utf-8") as f:
        # 读取第一行
        line = f.readline()
        json_obj = json.loads(line)
    
    # 格式化 Prompt
    prompt = prompt_format.format(**json_obj)
    
    # Tokenize
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    original_len = len(tokenized_prompt)
    print(f"Original Prompt Length: {original_len} tokens")
    
    # 截断逻辑 (参考 pred.py，但我们只取前 TARGET_LENGTH，为了保留连贯性)
    # pred.py 是取首尾，但对于做 KV Cache 可视化，连续的上下文可能更好理解
    # 我们这里修改一下：只取前 TARGET_LENGTH 个 token，确保 Prompt 完整性
    
    if original_len > TARGET_LENGTH:
        truncated_ids = tokenized_prompt[:TARGET_LENGTH]
        # 解码回文本
        final_prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        print(f"Truncated to {TARGET_LENGTH} tokens.")
    else:
        final_prompt = prompt
        print("Length is within limit, keeping original.")
        
    return final_prompt

if __name__ == "__main__":
    prompt_text = get_prompt()
    
    # 写入到 prompt_data.py
    # output_file = "aux_exp/prompt_data.py" # 错误路径：当前已经在 aux_exp 目录下
    output_file = "prompt_data.py" # 正确路径
    with open(output_file, "w", encoding="utf-8") as f:
        # 使用 repr() 安全地写入字符串，处理转义字符
        f.write(f"PROMPT_TEXT = {repr(prompt_text)}\n")
        
    print(f"Prompt saved to {output_file}")

