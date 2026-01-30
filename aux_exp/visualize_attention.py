import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import math

# ================= 配置区域 =================
# 如果您本地有模型路径，请修改这里。否则将尝试从 HuggingFace Hub 下载
MODEL_PATH = "/users/cyx/Meta-Llama-3-8B-Instruct" 

# Prompt: 设计包含明确的语义依赖
# 预期：最后一个 "France" 和 "is" 应该会去关注前面的 "Paris"
PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of France is"

# 要可视化的层级索引 (Llama-3-8B 有 32 层，索引 0-31)
TARGET_LAYERS = [0, 15, 31]
# ===========================================

def visualize_layer_attention(attention_tensor, layer_idx, tokens, output_dir="."):
    """
    绘制单层的 32 个 Head 的 Grid 图
    attention_tensor shape: (32, seq_len, seq_len)
    """
    num_heads = attention_tensor.shape[0]
    seq_len = attention_tensor.shape[1]
    
    # 设置 Grid 大小：4行8列 = 32个Head
    rows = 4
    cols = 8
    # 增加图片高度以容纳标签
    fig, axes = plt.subplots(rows, cols, figsize=(32, 20))
    fig.suptitle(f"Layer {layer_idx} Attention Maps (All 32 Heads)", fontsize=24)
    
    # 获取 Token 标签（为了显示清晰，稍微处理一下）
    # Llama3 的 token 可能包含特殊字符，替换掉以便显示
    display_tokens = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in tokens]

    print(f"Plotting Layer {layer_idx}...")

    # 简单的 Causal Mask 检查 (检查上三角是否全为0)
    # 注意：有时候会有极小值浮点误差，这里用一个很小的阈值判断
    upper_tri = torch.triu(attention_tensor[0], diagonal=1)
    is_causal = torch.all(upper_tri.abs() < 1e-6)
    print(f"Layer {layer_idx} Causal Mask Check (Head 0): {'Satisfied (Upper triangle is ~0)' if is_causal else 'Warning: Non-zero values in upper triangle'}")
    
    for i in range(num_heads):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 获取当前 Head 的 Attention Matrix
        attn_map = attention_tensor[i].float().cpu().numpy()
        
        # 绘制热力图
        sns.heatmap(
            attn_map,
            xticklabels=display_tokens, # 显示横轴标签
            yticklabels=display_tokens, # 显示纵轴标签
            cmap="viridis",    # 使用高对比度色阶
            cbar=True,         # 显示 Colorbar 以便区分数值高低
            square=True,
            ax=ax,
            vmin=0, vmax=1     # 固定色阶范围 0-1，方便跨头比较
        )
        ax.set_title(f"H{i}", fontsize=12)
        
        # 设置标签字体大小和旋转
        ax.set_xticklabels(display_tokens, rotation=90, fontsize=6)
        ax.set_yticklabels(display_tokens, rotation=0, fontsize=6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = os.path.join(output_dir, f"Llama3_Layer_{layer_idx}_Heads.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    print(f"Loading model: {MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # 显存充足，使用 bfloat16 (如果 GPU 支持) 或 float16
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            # 关键参数：开启 Attention 输出
            attn_implementation="eager" # 强制使用 Eager 模式以确保能拿到完整的 attention weights
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请确保您已登录 HuggingFace (huggingface-cli login) 或修改 MODEL_PATH 为本地路径。")
        return

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    print(f"Prompt Tokens: {tokens}")
    print("Running inference...")
    
    with torch.no_grad():
        # output_attentions=True 是抓取分数的关键
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions 是一个 tuple，包含每一层的 (batch, num_heads, seq, seq)
    all_attentions = outputs.attentions
    
    print(f"Total layers captured: {len(all_attentions)}")
    
    for layer_idx in TARGET_LAYERS:
        if layer_idx >= len(all_attentions):
            print(f"Layer {layer_idx} out of range, skipping.")
            continue
            
        # 取出该层数据: (batch, num_heads, seq, seq) -> (num_heads, seq, seq)
        # batch size 默认为 1
        layer_attn = all_attentions[layer_idx][0] 
        
        # 确保输出目录存在
        output_dir = "aux_exp/plot"
        os.makedirs(output_dir, exist_ok=True)
        visualize_layer_attention(layer_attn, layer_idx, tokens, output_dir=output_dir)

    print(f"Done! Check the generated PNG files in {output_dir}.")

if __name__ == "__main__":
    main()

