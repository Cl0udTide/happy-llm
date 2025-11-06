import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
    
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = f"{self.tokenizer.bos_token}{sample['completion']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # 1. 加载原始数据
        sample = json.loads(self.data[index])
        
        # 2. 根据 alpaca_gpt4_zh 的格式提取内容
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")
        
        # 3. 使用 Chat Template 格式化
        # 构造一个符合 apply_chat_template 输入格式的 messages 列表
        user_content = instruction
        if input_text:
            user_content += "\n" + input_text
            
        messages = [
            {"role": "user", "content": user_content}
        ]

        # --- 将用户部分和模型回答部分分开编码，这是计算 loss mask 的关键 ---
        
        # 3.1 编码用户部分 (Prompt)
        # add_special_tokens=False 确保我们能精确控制特殊 token
        prompt_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, add_special_tokens=False)

        # 3.2 编码模型回答部分 (Output)
        # 同样不加特殊 token
        output_ids = self.tokenizer(output_text, add_special_tokens=False)['input_ids']
        
        # 3.3 加上 EOS token
        # 根据 chat template, 每个 assistant 回答后都有一个 <|im_end|>
        output_ids.append(self.tokenizer.eos_token_id)
        
        # 4. 拼接成完整的 input_id
        input_id = prompt_ids + output_ids
        
        # --- 截断 ---
        input_id = input_id[:self.max_length]
        
        # 5. 构建 Loss Mask (更简单、更可靠的方法)
        # 只有 output_ids 部分需要计算 loss
        loss_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        loss_mask = loss_mask[:self.max_length]

        # 6. Padding
        padding_len = self.max_length - len(input_id)
        input_id = input_id + [self.padding] * padding_len
        loss_mask = loss_mask + [0] * padding_len
        
        # 7. 拆分 X 和 Y，并转换为 Tensor
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)