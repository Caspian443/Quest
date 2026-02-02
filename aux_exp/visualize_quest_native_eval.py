import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import sys
import types
from argparse import Namespace

# 引入项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 引入 Quest 评测模块
import evaluation.quest_attention
from evaluation.quest_attention import enable_quest_attention_eval

# ================= 配置 =================
MODEL_PATH = "/users/cyx/Meta-Llama-3-8B-Instruct" 
OUTPUT_DIR = "aux_exp/plot_quest_eval"
GENERATE_LEN = 30
TARGET_LAYERS = [2, 15, 31]
CHUNK_SIZE = 16
TOKEN_BUDGET = 128 # 激进预算

# 加载 Prompt
from prompt_data import PROMPT_TEXT
PROMPT = PROMPT_TEXT


# ================= 劫持逻辑 =================

# 全局日志
# LOGS 结构: List of items (按调用顺序)
# Item: {"attn": Tensor, "mask": Tensor, "layer_idx": int (inferred)}
GLOBAL_LOGS = []

# 保存原始函数
original_mask_func = evaluation.quest_attention.local_heavy_hitter_mask

# 计数器，用于推断 Layer ID
# Llama-3-8B 有 32 层。每次 Decode step 会调用 32 次 forward。
# 我们假设是顺序调用的 (Layer 0 -> Layer 1 -> ... -> Layer 31)
call_counter = 0
NUM_LAYERS = 32 

def mask_wrapper(attn_weights, token_budget, chunk_size):
    global call_counter
    
    # 1. 计算 Mask (调用原函数)
    mask = original_mask_func(attn_weights, token_budget, chunk_size)
    
    # 2. 推断 Layer ID
    # 注意：Prefill 阶段可能也会调用（如果 q_len > 1）
    # 但 evaluation/quest_attention.py:84 行写了: if q_len > 1: return flash_forward
    # 所以理论上 Quest Mask 逻辑只在 Decode (q_len=1) 且 layer >= 2 时触发？
    # 让我们看代码: `if q_len > 1 or self.layer_id < 2: return self.flash_forward`
    # 这意味着 Layer 0 和 Layer 1 是被跳过的！它们跑的是 flash_forward (全量)。
    # 只有 Layer 2 到 Layer 31 会跑 Quest 逻辑。
    
    # 这是一个重要的发现：Quest 默认策略是不处理前两层的。
    # 所以我们的 TARGET_LAYERS [0, 15, 31] 中，Layer 0 可能捕获不到数据。
    # 我们需要在画图时注意这一点。
    
    # 既然只有 layer >= 2 才调这里，我们可以维护一个内部计数器
    # 假设每次 step 会触发 (NUM_LAYERS - 2) 次调用
    
    # 但为了更稳健，我们可以尝试通过 inspect 栈帧来获取 self.layer_idx？
    # 或者简单点，我们只记录所有数据，画图时再人工对齐。
    
    # 捕获数据
    # attn_weights shape: (B, H, 1, SeqLen)
    # mask shape: (B, H, 1, SeqLen)
    
    # 我们只关心我们感兴趣的层
    # 这是一个粗略的估计，因为我们很难在函数内部知道外部的 self.layer_id
    # 除非我们也 Hook forward。
    
    # 方案修正：我们同时也 Patch `forward` 吗？
    # evaluation.quest_attention.forward 是一个函数，被绑定到实例方法上。
    # 我们可以 Patch `evaluation.quest_attention.forward` 函数本身！
    # 这样我们就能访问 `self.layer_idx` 了。
    
    return mask

# 更好的劫持方案：劫持 forward
original_forward = evaluation.quest_attention.forward

def forward_wrapper(self, hidden_states, *args, **kwargs):
    # 1. 调用原始 forward
    # 注意：原始 forward 会调用 local_heavy_hitter_mask
    # 我们依然需要劫持 mask 函数来拿到 mask，或者在 forward 返回后拿不到 mask (它是局部变量)
    
    # 看来必须要把 mask 函数劫持掉。
    # 我们可以利用一个全局变量 context 来传递 layer_idx
    
    global current_layer_idx
    # self.layer_id 是 Quest 代码里用的属性 (注意是 layer_id 不是 layer_idx)
    # 实际上在 enable_quest... 里设置了 model._modules[name].layer_id = layer_id
    if hasattr(self, 'layer_id'):
        current_layer_idx = self.layer_id
    elif hasattr(self, 'layer_idx'):
        current_layer_idx = self.layer_idx
    else:
        current_layer_idx = -1
        
    return original_forward(self, hidden_states, *args, **kwargs)

# 全局变量用于通信
current_layer_idx = -1

def mask_wrapper_with_logging(attn_weights, token_budget, chunk_size):
    mask = original_mask_func(attn_weights, token_budget, chunk_size)
    
    # 记录
    if current_layer_idx in TARGET_LAYERS:
        # 存下来
        # attn_weights 是传进来的 "Quantized Weight" (Q @ K_chunk_max)
        # 这其实不是 Full Attention，而是 Quest 的估算分数。
        # 但这正是我们要可视化的：Quest 依据什么做出的决定。
        
        GLOBAL_LOGS.append({
            "layer": current_layer_idx,
            "score": attn_weights.detach().cpu(),
            "mask": mask.detach().cpu()
        })
        
    return mask

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
    
    # 1. 启用 Quest
    print("Enabling Quest Attention...")
    args = Namespace(token_budget=TOKEN_BUDGET, chunk_size=CHUNK_SIZE)
    enable_quest_attention_eval(model, args)
    
    # 2. 应用劫持
    print("Applying Hijack...")
    evaluation.quest_attention.local_heavy_hitter_mask = mask_wrapper_with_logging
    # 同时也替换 forward 以便传递 layer_id
    evaluation.quest_attention.forward = forward_wrapper
    # 重新绑定 method (因为 enable_quest... 已经绑定过了，我们需要更新绑定)
    # 我们需要重新跑一遍 enable_quest? 不，直接修改 bound method 比较难。
    # 简单粗暴：我们再次手动遍历 model modules 重新绑定我们的 wrapper
    
    from transformers.models.llama.modeling_llama import LlamaAttention
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention):
            # 重新绑定 forward，指向我们的 wrapper
            # 注意 wrapper 第一个参数是 self
            module.forward = types.MethodType(forward_wrapper, module)

    # 3. 运行推理
    print("Running Inference...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # Prefill (不会触发 Quest 逻辑，或者触发 flash_forward)
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
    
    # 4. 可视化
    print(f"Captured {len(GLOBAL_LOGS)} log entries.")
    
    # 按层整理
    layer_logs = {l: [] for l in TARGET_LAYERS}
    for item in GLOBAL_LOGS:
        layer_logs[item['layer']].append(item)
        
    for layer_idx in TARGET_LAYERS:
        history = layer_logs[layer_idx]
        if not history:
            print(f"No logs for Layer {layer_idx} (Maybe skipped by Quest?)")
            continue
            
        print(f"Plotting Layer {layer_idx} ({len(history)} steps)...")
        
        num_steps = len(history)
        # 注意：attn_weights 是 (B, H, 1, SeqLen)
        # SeqLen 是 Chunk 后的长度还是原始长度？
        # 看代码：mask_bottom 是被还原回 seq_length 的 (line 67)
        # attn_weights (传入 mask 函数的) 是 quantized_weight，形状是 (B, H, 1, ChunkNum)
        # 哎呀，这就麻烦了。传入 mask 函数的是 quantized_weight。
        # 它还没有被 expand 回 seq_len。
        # 只有返回的 mask 是被 expand 后的。
        
        # 我们来看看 mask 的形状
        # 修正：序列长度会随时间增长，我们需要取最大长度（最后一步的长度）来初始化数组
        final_mask = history[-1]['mask'] 
        max_len = final_mask.shape[-1]
        
        rows = 4
        cols = 8
        fig, axes = plt.subplots(rows, cols, figsize=(32, 20))
        fig.suptitle(f"Layer {layer_idx} Quest Evaluation (Green=Selected)", fontsize=24)
        
        for h in range(32):
            row = h // cols
            col = h % cols
            ax = axes[row, col]
            
            # 画 Mask
            # (Steps, MaxSeqLen)
            mask_arr = np.zeros((num_steps, max_len))
            
            for step, item in enumerate(history):
                # item['mask'] shape: (B, H, 1, SeqLen)
                m = item['mask'][0, h, 0, :].float().numpy()
                current_len = len(m)
                mask_arr[step, :current_len] = m
                
            # 画图
            # 底色：黑色
            # 选中：绿色
            
            # 这里我们没有 Full Attention 作为底图（因为 Quest 逻辑里只算了 Approximate Score）
            # 我们就只画 Mask 分布，看看它是怎么跳的
            
            sns.heatmap(
                mask_arr,
                cmap="Greens", # 0=White/Light, 1=Green
                cbar=False,
                ax=ax,
                vmin=0, vmax=1
            )
            
            ax.set_title(f"H{h}", fontsize=12)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_path = os.path.join(OUTPUT_DIR, f"Quest_Eval_Layer_{layer_idx}.png")
        plt.savefig(output_path, dpi=150)
        print(f"Saved {output_path}")
        plt.close()

if __name__ == "__main__":
    main()

