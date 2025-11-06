import os
import torch
import argparse
from transformers import AutoTokenizer

from model import Transformer, ModelConfig

def find_pth_file(path: str) -> str:
    """
    在指定路径下查找 .pth 文件。
    - 如果 path 是文件，直接返回。
    - 如果 path 是目录，查找该目录下唯一的 .pth 文件。
    - 如果找不到或找到多个，则抛出异常。
    """
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
        
    if os.path.isdir(path):
        pth_files = [f for f in os.listdir(path) if f.endswith(".pth")]
        if len(pth_files) == 1:
            return os.path.join(path, pth_files[0])
        elif len(pth_files) == 0:
            raise FileNotFoundError(f"在目录 '{path}' 中没有找到 .pth 文件。")
        else:
            raise ValueError(f"在目录 '{path}' 中找到多个 .pth 文件，请明确指定一个：{pth_files}")
    
    raise FileNotFoundError(f"路径 '{path}' 不是一个有效的 .pth 文件或包含 .pth 文件的目录。")

def generate_text(
    model_path: str, 
    prompt: str, 
    tokenizer_path: str = "./tokenizer_k/",
    max_new_tokens: int = 50,
    temperature: float = 0.7
):
    """
    加载微缩模型并生成文本。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- 使用设备: {device} ---")

    # --- 1. 定义固定模型配置 ---
    lm_config = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=8192,
        max_seq_len=256,
    )
    print("--- 使用固定的模型配置 ---")

    # --- 2. 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"--- 从 '{tokenizer_path}' 加载 Tokenizer 成功 ---")

    # --- 3. 初始化模型并加载权重 ---
    try:
        model_path = find_pth_file(model_path)
        print(f"--- 自动找到权重文件: {model_path} ---")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return
    
    model = Transformer(lm_config).to(device)
    state_dict = torch.load(model_path, map_location=device)        
    model.load_state_dict(state_dict)
    model.eval() # 切换到评估模式
    print(f"--- 从 '{model_path}' 加载模型权重成功 ---")

    # --- 4. 编码 Prompt 并生成 ---
    print("\n--- 开始生成 ---")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    # --- 5. 解码并打印结果 ---
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print("\n--- 完整输出 (包含Prompt) ---")
    print(generated_text)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LLM Generation Script")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的 .pth 模型权重文件路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入的提示词")
    args = parser.parse_args()

    generate_text(model_path=args.model_path, prompt=args.prompt)