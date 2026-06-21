#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将现有数据集转换为HMTL格式
支持训练集和评估集
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "02_模型代码")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from hmtl_utils import get_hmtl_labels, print_emotion_distribution

def convert_eval_set():
    """转换评估集"""
    input_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json")
    output_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set_hmtl.json")
    
    print("="*60)
    print("转换评估集到HMTL格式")
    print("="*60)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hmtl_data = []
    label_4_counts = Counter()
    label_3_counts = Counter()
    
    for item in data:
        text = item.get('text', '')
        emotion = item.get('main_emotion', '平静')
        
        if not text:
            continue
        
        # 获取HMTL标签
        labels = get_hmtl_labels(emotion)
        
        hmtl_item = {
            'text': text,
            'original_emotion': emotion,
            'original_polarity': item.get('polarity', ''),
            'label_4': labels['label_4'],
            'label_3': labels['label_3'],
            'arousal': labels['arousal'],
            'valence': labels['valence'],
            'source': item.get('source', 'eval_set'),
            'id': item.get('id', '')
        }
        hmtl_data.append(hmtl_item)
        
        label_4_counts[labels['label_4']] += 1
        label_3_counts[labels['label_3']] += 1
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hmtl_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已转换 {len(hmtl_data)} 条评估数据")
    print(f"✓ 保存到: {output_path}")
    
    # 打印分布
    print_emotion_distribution(label_4_counts, label_3_counts)
    
    return output_path


def convert_training_set():
    """转换训练集（balanced_training_samples.json）"""
    input_path = r"d:\silent like onion\03_数据\balanced_training_samples.json"
    output_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json")
    
    print("\n" + "="*60)
    print("转换训练集到HMTL格式")
    print("="*60)
    
    if not os.path.exists(input_path):
        print(f"✗ 未找到训练集: {input_path}")
        return None
    
    # 读取原始数据（处理前缀问题）
    with open(input_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    # 处理可能的前缀（如"yy{"）
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]
    
    data = json.loads(raw)
    samples = data.get("samples", [])
    labels = data.get("labels", [])
    
    if len(samples) != len(labels):
        print(f"[WARNING] samples与labels数量不匹配: {len(samples)} vs {len(labels)}")
    
    hmtl_data = []
    label_4_counts = Counter()
    label_3_counts = Counter()
    
    n = min(len(samples), len(labels))
    for i in range(n):
        text = samples[i]
        emotion = labels[i]
        
        if not text or not emotion:
            continue
        
        # 获取HMTL标签
        hmtl_labels = get_hmtl_labels(emotion)
        
        hmtl_item = {
            'text': text,
            'original_emotion': emotion,
            'label_4': hmtl_labels['label_4'],
            'label_3': hmtl_labels['label_3'],
            'arousal': hmtl_labels['arousal'],
            'valence': hmtl_labels['valence'],
            'source': 'balanced_training_samples',
            'id': f'train_{i:05d}'
        }
        hmtl_data.append(hmtl_item)
        
        label_4_counts[hmtl_labels['label_4']] += 1
        label_3_counts[hmtl_labels['label_3']] += 1
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hmtl_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 已转换 {len(hmtl_data)} 条训练数据")
    print(f"✓ 保存到: {output_path}")
    
    # 打印分布
    print_emotion_distribution(label_4_counts, label_3_counts)
    
    return output_path


def main():
    """主函数"""
    print("HMTL数据集转换工具\n")
    
    # 转换评估集
    eval_path = convert_eval_set()
    
    # 转换训练集
    train_path = convert_training_set()
    
    print("\n" + "="*60)
    print("✅ 数据转换完成！")
    print("="*60)
    print(f"\n生成的文件:")
    if eval_path:
        print(f"  评估集: {eval_path}")
    if train_path:
        print(f"  训练集: {train_path}")
    
    print("\n下一步:")
    print("  python hmtl_train.py  # 开始训练")


if __name__ == "__main__":
    main()
