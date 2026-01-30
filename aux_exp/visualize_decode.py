import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np

# ================= 配置区域 =================
MODEL_PATH = "/users/cyx/Meta-Llama-3-8B-Instruct" 
OUTPUT_DIR = "aux_exp/plot_decode"

# 构造一个稍长的 Prompt，包含多个事实，最后进行提问，诱导模型生成长一点的回答
# 我们希望模型在生成回答的过程中，能够不断回看上文的线索
PROMPT = """Review: The movie "Interstellar" is a masterpiece of visual effects and storytelling. 
Director Christopher Nolan has outdone himself. The acting by Matthew McConaughey is stellar, bringing deep emotion to the character of Cooper.
However, the plot can be a bit confusing at times, especially with the time dilation concepts.
Despite this, the soundtrack by Hans Zimmer is absolutely breathtaking and elevates the entire experience.

Question: Summarize the pros and cons mentioned in the review and give a final verdict.

Answer: The review highlights several strong points about "Interstellar"."""

# 生成长度：稍微长一点，以便观察 Decode 阶段的变化
GENERATE_LEN = 30

# 要可视化的层级索引
TARGET_LAYERS = [0, 15, 31]
# ===========================================

def visualize_full_attention(attention_logs, layer_idx, all_tokens, output_dir):
    """
    绘制完整的 Attention 历史 (Prefill + Decode)
    attention_logs: List 
      - 第0个元素: Prefill 矩阵 (num_heads, prompt_len, prompt_len)
      - 后续元素: Decode 向量 (num_heads, 1, current_len)
    """
    prefill_matrix = attention_logs[0] # (H, P, P)
    num_heads = prefill_matrix.shape[0]
    prompt_len = prefill_matrix.shape[1]
    
    decode_steps = attention_logs[1:]
    num_generated = len(decode_steps)
    total_len = prompt_len + num_generated
    
    # 准备画布
    rows = 4
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32)) # 正方形大图
    fig.suptitle(f"Layer {layer_idx} Full Attention (Prefill + Decode)", fontsize=24)
    
    # 清洗 Token 标签
    display_tokens = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in all_tokens]

    print(f"Plotting Layer {layer_idx} Full Map ({total_len}x{total_len})...")
    
    for h in range(num_heads):
        # 构建完整的大矩阵 (Total x Total)
        # 初始化为 0 (Masked 区域)
        full_heatmap = np.zeros((total_len, total_len))
        
        # 1. 填入 Prefill 左上角
        # prefill_matrix[h] 是 (P, P)
        full_heatmap[:prompt_len, :prompt_len] = prefill_matrix[h].float().numpy()
        
        # 2. 填入 Decode 行
        for i, step_attn in enumerate(decode_steps):
            # step_attn shape: (num_heads, 1, current_len)
            # current_len = prompt_len + i + 1
            current_len = step_attn.shape[-1]
            vector = step_attn[h, 0, :].float().numpy()
            
            # 填入第 (prompt_len + i) 行
            full_heatmap[prompt_len + i, :current_len] = vector
            
        row = h // cols
        col = h % cols
        ax = axes[row, col]
        
        # 绘制
        sns.heatmap(
            full_heatmap,
            xticklabels=False, 
            yticklabels=False,
            cmap="viridis",
            cbar=False, # 小图就不加bar了，太乱
            square=True,
            ax=ax,
            vmin=0, vmax=1
        )
        ax.set_title(f"H{h}", fontsize=12)
        
        # 绘制分割线，区分 Prompt 和 Generated
        # 加粗虚线，使用显眼的红色，并添加文字标注
        ax.axhline(y=prompt_len, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axvline(x=prompt_len, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        
        # 仅在第一张子图(H0)添加文字说明，避免太乱
        if h == 0:
            # 在图内左上角添加 "Prompt"
            ax.text(1, 1, 'Prompt Region', color='white', fontsize=8, ha='left', va='top', fontweight='bold')
            # 在分割线下方添加 "Decode Start"
            ax.text(1, prompt_len + 2, 'Decode Region', color='white', fontsize=8, ha='left', va='top', fontweight='bold')

        # 同样，只在边缘显示标签
        if row == rows-1:
            ax.set_xticks(np.arange(len(display_tokens)) + 0.5)
            ax.set_xticklabels(display_tokens, rotation=90, fontsize=4)
        if col == 0:
            ax.set_yticks(np.arange(len(display_tokens)) + 0.5)
            ax.set_yticklabels(display_tokens, rotation=0, fontsize=4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path = os.path.join(output_dir, f"Llama3_Full_Layer_{layer_idx}.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager" 
    )

    # 1. Prefill 阶段
    print("Running Prefill...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    prompt_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # 第一次前向传播，获取 KV Cache
    # 注意：必须要加上 output_attentions=True，否则 outputs.attentions 是 None
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=True)
    
    past_key_values = outputs.past_key_values
    input_ids = inputs.input_ids
    
    # 记录生成的 Token
    generated_ids = []
    
    # 存储 Attention 历史
    # 结构: {layer_idx: [full_matrix]}
    # 我们将 Prefill 的矩阵作为起点，然后每一步把新的一行拼上去
    attention_logs = {l: [] for l in TARGET_LAYERS}

    # 保存 Prefill 阶段的 Attention
    # outputs.attentions[layer][0] shape: (num_heads, seq_len, seq_len)
    for layer_idx in TARGET_LAYERS:
        # 取出 Prefill 矩阵
        prefill_attn = outputs.attentions[layer_idx][0].cpu() # (num_heads, prompt_len, prompt_len)
        attention_logs[layer_idx].append(prefill_attn)
    
    print(f"Starting Decode (Target: {GENERATE_LEN} tokens)...")
    
    # 2. Decode 循环
    curr_input_ids = input_ids[:, -1:] 
    
    for step in range(GENERATE_LEN):
        with torch.no_grad():
            outputs = model(
                curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        curr_input_ids = next_token_id
        generated_ids.append(next_token_id.item())
        
        # 捕获 Attention 并拼接
        for layer_idx in TARGET_LAYERS:
            # Decode Step 的 attention shape: (num_heads, 1, current_total_len)
            step_attn = outputs.attentions[layer_idx][0].cpu() 
            attention_logs[layer_idx].append(step_attn)
            
        print(f"Step {step+1}/{GENERATE_LEN}: Generated token ID {next_token_id.item()}")

    # 转换生成的 Token ID 为文本
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
    all_tokens = prompt_tokens + generated_tokens
    print(f"Generated text: {tokenizer.decode(generated_ids)}")
    
    # 3. 可视化
    for layer_idx in TARGET_LAYERS:
        visualize_full_attention(
            attention_logs[layer_idx], 
            layer_idx, 
            all_tokens,
            OUTPUT_DIR
        )

    print(f"All done! Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

