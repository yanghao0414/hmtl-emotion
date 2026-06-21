#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证HMTL Dataloader输出
检查batch的shape和dtype
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "02_模型代码")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
from transformers import BertTokenizer
from hmtl_dataset import create_dataloaders

def verify_dataloader():
    """验证Dataloader输出格式"""
    print("="*60)
    print("HMTL Dataloader 验证")
    print("="*60)
    
    # 数据路径
    train_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json")
    eval_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set_hmtl.json")
    
    # 创建tokenizer和dataloaders
    print("\n加载数据...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_loader, eval_loader = create_dataloaders(
        train_path, eval_path, tokenizer,
        batch_size=16, max_length=128
    )
    
    # 获取一个batch
    print("\n获取一个训练batch...")
    batch = next(iter(train_loader))
    
    print("\n" + "="*60)
    print("Batch 张量信息")
    print("="*60)
    
    # 按照用户要求的格式输出
    print("\n【输入张量】")
    print(f"input_ids.shape     : {batch['input_ids'].shape}, dtype: {batch['input_ids'].dtype}")
    print(f"attention_mask.shape: {batch['attention_mask'].shape}, dtype: {batch['attention_mask'].dtype}")
    
    print("\n【标签张量（对应用户要求的命名）】")
    # 用户要求的命名: labels_4, labels_3, true_A, true_V
    print(f"labels_4 (label_4).shape : {batch['label_4'].shape}, dtype: {batch['label_4'].dtype}")
    print(f"labels_3 (label_3).shape : {batch['label_3'].shape}, dtype: {batch['label_3'].dtype}")
    print(f"true_A (arousal).shape   : {batch['arousal'].shape}, dtype: {batch['arousal'].dtype}")
    print(f"true_V (valence).shape   : {batch['valence'].shape}, dtype: {batch['valence'].dtype}")
    
    print("\n【详细说明】")
    print(f"Batch size: {batch['label_4'].shape[0]}")
    print(f"Sequence length: {batch['input_ids'].shape[1]}")
    
    print("\n【标签值范围】")
    print(f"label_4 范围: {batch['label_4'].min().item()} - {batch['label_4'].max().item()} (期望: 0-3)")
    print(f"label_3 范围: {batch['label_3'].min().item()} - {batch['label_3'].max().item()} (期望: 0-2)")
    print(f"arousal 范围: {batch['arousal'].min().item():.3f} - {batch['arousal'].max().item():.3f} (期望: 0-1)")
    print(f"valence 范围: {batch['valence'].min().item():.3f} - {batch['valence'].max().item():.3f} (期望: -1 to 1)")
    
    # 显示一个样本
    print("\n" + "="*60)
    print("示例样本 (batch中的第1个)")
    print("="*60)
    print(f"label_4 (4分类): {batch['label_4'][0].item()} ", end="")
    label_4_names = {0: '积极', 1: '激活型消极', 2: '非激活型消极', 3: '平静'}
    print(f"({label_4_names.get(batch['label_4'][0].item(), '未知')})")
    
    print(f"label_3 (3分类): {batch['label_3'][0].item()} ", end="")
    label_3_names = {0: '积极', 1: '消极', 2: '平静'}
    print(f"({label_3_names.get(batch['label_3'][0].item(), '未知')})")
    
    print(f"arousal (唤醒度): {batch['arousal'][0].item():.3f}")
    print(f"valence (效价): {batch['valence'][0].item():.3f}")
    
    print("\n" + "="*60)
    print("✅ Dataloader验证通过!")
    print("="*60)
    
    return batch


def verify_mapping_distribution():
    """验证4核心分类的映射分布"""
    import json
    from collections import Counter
    
    print("\n" + "="*60)
    print("4核心分类映射分布验证")
    print("="*60)
    
    # 加载转换后的数据
    with open(str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json"), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 统计原始情绪和4分类的对应关系
    emotion_to_label4 = {}
    label4_counts = Counter()
    
    for item in train_data:
        orig_emo = item['original_emotion']
        label_4 = item['label_4']
        
        if orig_emo not in emotion_to_label4:
            emotion_to_label4[orig_emo] = label_4
        
        label4_counts[label_4] += 1
    
    # 按4分类分组显示
    print("\n【4核心分类的映射关系】")
    label_4_names = {
        0: '积极 (Positive)',
        1: '激活型消极 (High-A Neg.)',
        2: '低落型消极 (Low-A Neg.)',
        3: '平静 (Neutral)'
    }
    
    for label_id in sorted(emotion_to_label4.values()):
        emotions = [emo for emo, lid in emotion_to_label4.items() if lid == label_id]
        count = label4_counts[label_id]
        pct = count / len(train_data) * 100
        
        print(f"\n[{label_id}] {label_4_names[label_id]}: {count}条 ({pct:.1f}%)")
        print(f"    包含情绪: {', '.join(sorted(emotions))}")
    
    print("\n【A/V维度映射示例】")
    # 显示几个代表性情绪的A/V值
    sample_emotions = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
    
    from hmtl_utils import EMOTION_MAP
    
    print("\n情绪 -> [4分类, 3分类, Arousal, Valence]")
    print("-" * 60)
    for emo in sample_emotions:
        if emo in EMOTION_MAP:
            mapping = EMOTION_MAP[emo]
            print(f"{emo:4s} -> [{mapping['label_4']}, {mapping['label_3']}, "
                  f"A={mapping['arousal']:.1f}, V={mapping['valence']:+.1f}]")


if __name__ == "__main__":
    # 验证Dataloader
    batch = verify_dataloader()
    
    # 验证映射分布
    verify_mapping_distribution()
