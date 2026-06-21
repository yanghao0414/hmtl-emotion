#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL数据集类
支持PyTorch DataLoader
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict


class HMTLEmotionDataset(Dataset):
    """HMTL情绪数据集"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer: BertTokenizer,
                 max_length: int = 128):
        """
        Args:
            data_path: HMTL格式的JSON数据路径
            tokenizer: BERT分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"✓ 加载了 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            dict: {
                'input_ids': [max_length],
                'attention_mask': [max_length],
                'label_4': int,
                'label_3': int,
                'arousal': float,
                'valence': float
            }
        """
        item = self.data[idx]
        text = item['text']
        
        # BERT编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # [max_length]
            'label_4': torch.tensor(item['label_4'], dtype=torch.long),
            'label_3': torch.tensor(item['label_3'], dtype=torch.long),
            'arousal': torch.tensor(item['arousal'], dtype=torch.float),
            'valence': torch.tensor(item['valence'], dtype=torch.float)
        }


def create_dataloaders(
    train_path: str,
    eval_path: str,
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> tuple:
    """
    创建训练和评估数据加载器
    
    Returns:
        (train_loader, eval_loader)
    """
    # 创建数据集
    train_dataset = HMTLEmotionDataset(train_path, tokenizer, max_length)
    eval_dataset = HMTLEmotionDataset(eval_path, tokenizer, max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, eval_loader


def test_dataset():
    """测试数据集"""
    print("测试HMTL数据集...")
    
    # 创建tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建测试数据
    test_data = [
        {
            'text': '我很开心',
            'label_4': 0,
            'label_3': 0,
            'arousal': 0.7,
            'valence': 0.9
        },
        {
            'text': '我很生气',
            'label_4': 1,
            'label_3': 1,
            'arousal': 0.9,
            'valence': -0.8
        }
    ]
    
    # 保存测试数据
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parents[3]
    test_path = str(_PROJECT_ROOT / '05_数据文件' / 'test_hmtl.json')
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    
    # 创建数据集
    dataset = HMTLEmotionDataset(test_path, tokenizer)
    
    # 测试获取样本
    sample = dataset[0]
    print(f"\n✓ 数据集测试通过!")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  label_4: {sample['label_4']}")
    print(f"  label_3: {sample['label_3']}")
    print(f"  arousal: {sample['arousal']:.2f}")
    print(f"  valence: {sample['valence']:.2f}")
    
    # 测试DataLoader
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    print(f"\n✓ DataLoader测试通过!")
    print(f"  batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  batch label_4 shape: {batch['label_4'].shape}")


if __name__ == "__main__":
    test_dataset()
