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
    fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
    fig.suptitle(f"Layer {layer_idx} Attention Maps (All 32 Heads)", fontsize=20)
    
    # 获取 Token 标签（为了显示清晰，稍微处理一下）
    # Llama3 的 token 可能包含特殊字符，替换掉以便显示
    display_tokens = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in tokens]

    print(f"Plotting Layer {layer_idx}...")
    
    for i in range(num_heads):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 获取当前 Head 的 Attention Matrix
        attn_map = attention_tensor[i].float().cpu().numpy()
        
        # 绘制热力图
        # vmin=0, vmax=1 实际上大多数 attention 非常稀疏，用 log scale 可能更好，
        # 但为了直观看到“强关注”，这里不做 log 处理，让强点更突出。
        sns.heatmap(
            attn_map,
            xticklabels=False, # 小图就不显示横轴标签了，太挤
            yticklabels=False, # 小图就不显示纵轴标签了
            cmap="viridis",    # 使用高对比度色阶
            cbar=False,        # 不显示 Colorbar 节省空间
            square=True,
            ax=ax
        )
        ax.set_title(f"H{i}", fontsize=10)
        
        # 只在左下角的图显示标签，或者干脆都不显示，太密了看不清
        # 如果只想看几个特定的，可以单独画。这里为了看分布，先略去标签。

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        
        visualize_layer_attention(layer_attn, layer_idx, tokens)

    print("Done! Check the generated PNG files.")

if __name__ == "__main__":
    main()

