import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import sys
import types
from argparse import Namespace
import math

# 引入项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 引入 Quest 评测模块
import evaluation.quest_attention
from evaluation.quest_attention import enable_quest_attention_eval

# ================= 配置 =================
MODEL_PATH = "/users/cyx/Meta-Llama-3-8B-Instruct" 
OUTPUT_DIR = "aux_exp/plot_quest_native"
GENERATE_LEN = 30
TARGET_LAYERS = [2, 15, 31]
CHUNK_SIZE = 16
TOKEN_BUDGET = 128 

# 加载 Prompt
try:
    from prompt_data import PROMPT_TEXT
    PROMPT = PROMPT_TEXT
except ImportError:
    PROMPT = "The capital of France is Paris. " * 50

# ================= 数据结构 =================
GLOBAL_DATA = {}

# ================= 极简 Wrapper (只负责偷数据) =================

def create_visualization_wrapper(original_forward):
    def wrapper(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        # 强制开启 output_attentions 以便 Quest 返回分数
        output_attentions = True
        
        # 调用原始 Quest Forward
        # 显式传递参数，避免 *args / **kwargs 混淆
        # 注意：original_forward 是一个 Bound Method，不需要传 self
        attn_output, attn_weights, past_key_value = original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        # 记录数据
        # self.layer_id 是 Quest 代码注入的属性
        layer_idx = getattr(self, 'layer_id', -1)
        
        if layer_idx in TARGET_LAYERS:
            if layer_idx not in GLOBAL_DATA:
                GLOBAL_DATA[layer_idx] = {"prefill": None, "decode": []}
                
            # 判断是 Prefill 还是 Decode
            # hidden_states: (B, Seq, Dim)
            q_len = hidden_states.shape[1]
            
            # Quest 返回的 attn_weights 已经是 Softmax 后的概率
            
            if attn_weights is not None:
                if q_len > 1: # Prefill
                    GLOBAL_DATA[layer_idx]["prefill"] = attn_weights.detach().cpu()
                else: # Decode
                    GLOBAL_DATA[layer_idx]["decode"].append(attn_weights.detach().cpu())
                
        return attn_output, attn_weights, past_key_value
        
    return wrapper

# ================= 可视化函数 (保持不变) =================

def visualize_layer(layer_idx, data):
    print(f"Visualizing Layer {layer_idx}...")
    
    prefill_probs = data["prefill"]
    decode_steps = data["decode"] 
    
    if prefill_probs is None:
        print(f"No prefill data for Layer {layer_idx}")
        return

    num_heads = prefill_probs.shape[1]
    rows = 4
    cols = 8
    
    S_pre = prefill_probs.shape[-1]
    S_gen = len(decode_steps)
    S_total = S_pre + S_gen
    
    fig, axes = plt.subplots(rows, cols, figsize=(40, 24))
    fig.suptitle(f"Layer {layer_idx} Quest Native Vis (Selected Pages Highlighted)", fontsize=24)
    
    for h in range(num_heads):
        row = h // cols
        col = h % cols
        ax = axes[row, col]
        
        full_matrix = np.zeros((S_total, S_total))
        
        # 1. Prefill
        p_attn = prefill_probs[0, h, :, :].float().numpy()
        full_matrix[:S_pre, :S_pre] = p_attn
        
        # 2. Decode
        for i, step_prob in enumerate(decode_steps):
            prob = step_prob[0, h, 0, :].float().numpy()
            row_idx = S_pre + i
            current_len = len(prob)
            full_matrix[row_idx, :current_len] = prob
            
        # Draw
        display_matrix = np.power(full_matrix, 0.3)
        
        sns.heatmap(
            display_matrix,
            cmap="magma",
            cbar=False,
            ax=ax,
            vmin=0, vmax=0.5
        )
        
        ax.axhline(y=S_pre, color='white', linestyle='--', linewidth=0.5)
        ax.axvline(x=S_pre, color='white', linestyle='--', linewidth=0.5)
        ax.set_title(f"H{h}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"Quest_Native_Layer_{layer_idx}.png")
    plt.savefig(save_path, dpi=100)
    print(f"Saved {save_path}")
    plt.close()

# ================= 主流程 =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 1. Enable Quest
    # 这步操作会把 model 里的 forward 替换成 evaluation.quest_attention.forward
    print("Enabling Quest Attention...")
    args = Namespace(token_budget=TOKEN_BUDGET, chunk_size=CHUNK_SIZE)
    enable_quest_attention_eval(model, args)
    
    # 2. Inject Visualization Wrapper
    print("Injecting Visualization Wrapper...")
    from transformers.models.llama.modeling_llama import LlamaAttention
    
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # 获取已经被 Quest 替换过的 forward
            quest_forward = module.forward
            # 用我们的 wrapper 包裹它
            module.forward = types.MethodType(create_visualization_wrapper(quest_forward), module)

    # 3. Inference
    print("Running Inference...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    print(f"Prefill ({input_ids.shape[-1]} tokens)...")
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    
    curr_input_ids = input_ids[:, -1:]
    generated_ids = []
    
    print("Start Decoding...")
    for step in range(GENERATE_LEN):
        with torch.no_grad():
            outputs = model(curr_input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        curr_input_ids = next_id
        generated_ids.append(next_id.item())
        print(f"Step {step}: {next_id.item()}")
        
    print(f"Text: {tokenizer.decode(generated_ids)}")
    
    # 4. Visualize
    for layer_idx in TARGET_LAYERS:
        if layer_idx in GLOBAL_DATA:
            visualize_layer(layer_idx, GLOBAL_DATA[layer_idx])

if __name__ == "__main__":
    main()
