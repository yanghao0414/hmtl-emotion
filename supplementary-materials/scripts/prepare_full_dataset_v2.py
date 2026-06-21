#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL数据集预处理 V2 - 按对话文件划分，防止数据泄露
同一个对话的所有句子只会出现在训练集或评估集中
"""

import json
import os
import random
from collections import Counter, defaultdict

INPUT_DIR = r'd:\silent like onion\03_数据\文本已标注数据'
OUTPUT_DIR = r'd:\bigcreate\05_数据文件'
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

# 标签映射（与V1相同）
EMOTION_7_MAP = {
    '愤怒': 0, '焦虑': 1, '快乐': 2, '悲伤': 3,
    '失望': 4, '支持': 5, '平静': 6,
    '生气': 0,
    '紧张': 1, '担心': 1, '害怕': 1, '恐惧': 1, '困惑': 1, '犹豫': 1,
    '兴奋': 2, '激动': 2, '希望': 2, '期待': 2, '自信': 2,
    '沮丧': 3, '无助': 3,
    '理解': 5, '安慰': 5, '鼓励': 5,
    '放松': 6, '坚定': 6,
    '感激': 2, '果断': 6,
}

LABEL_7_TO_4 = {0:1, 1:1, 2:0, 3:2, 4:2, 5:0, 6:3}
LABEL_4_TO_3 = {0:0, 1:1, 2:1, 3:2}

AROUSAL_MAP = {0:0.85, 1:0.75, 2:0.70, 3:0.30, 4:0.35, 5:0.50, 6:0.25}
VALENCE_MAP = {0:-0.80, 1:-0.60, 2:0.75, 3:-0.70, 4:-0.50, 5:0.60, 6:0.05}


def process_files():
    """按对话文件组织数据"""
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    # 按文件分组：{filename: [samples]}
    file_samples = {}
    skipped = 0
    
    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            data = json.load(open(fpath, 'r', encoding='utf-8'))
        except Exception:
            skipped += 1
            continue
        
        if not isinstance(data, list):
            skipped += 1
            continue
        
        samples = []
        for msg in data:
            if msg.get('role') != 'client' or 'annotation' not in msg:
                continue
            for ann in msg['annotation']:
                sentence = ann.get('sentence', '').strip()
                label_name = ann.get('label', '').strip()
                if not sentence or not label_name or len(sentence) < 2:
                    continue
                if label_name not in EMOTION_7_MAP:
                    skipped += 1
                    continue
                
                label_7 = EMOTION_7_MAP[label_name]
                label_4 = LABEL_7_TO_4[label_7]
                label_3 = LABEL_4_TO_3[label_4]
                
                noise_a = random.uniform(-0.08, 0.08)
                noise_v = random.uniform(-0.08, 0.08)
                arousal = max(0.0, min(1.0, AROUSAL_MAP[label_7] + noise_a))
                valence = max(-1.0, min(1.0, VALENCE_MAP[label_7] + noise_v))
                
                samples.append({
                    'text': sentence,
                    'original_emotion': label_name,
                    'label_7': label_7,
                    'label_4': label_4,
                    'label_3': label_3,
                    'arousal': round(arousal, 4),
                    'valence': round(valence, 4),
                    'source': fname,
                })
        
        if samples:
            file_samples[fname] = samples
    
    print(f"对话文件数: {len(file_samples)}")
    print(f"总句子数: {sum(len(v) for v in file_samples.values())}")
    print(f"跳过: {skipped}")
    
    return file_samples


def split_by_file(file_samples):
    """按对话文件划分训练集和评估集（同一对话不会被拆分）"""
    random.seed(RANDOM_SEED)
    
    filenames = list(file_samples.keys())
    random.shuffle(filenames)
    
    split_idx = int(len(filenames) * TRAIN_RATIO)
    train_files = filenames[:split_idx]
    eval_files = filenames[split_idx:]
    
    train_data = []
    for fname in train_files:
        for i, s in enumerate(file_samples[fname]):
            s['id'] = f"{fname}_{i}"
            train_data.append(s)
    
    eval_data = []
    for fname in eval_files:
        for i, s in enumerate(file_samples[fname]):
            s['id'] = f"{fname}_{i}"
            eval_data.append(s)
    
    # 验证无泄露
    train_sources = set(s['source'] for s in train_data)
    eval_sources = set(s['source'] for s in eval_data)
    overlap = train_sources & eval_sources
    assert len(overlap) == 0, f"数据泄露！重叠文件: {len(overlap)}"
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, 'full_training_set_hmtl_v2.json')
    eval_path = os.path.join(OUTPUT_DIR, 'full_eval_set_hmtl_v2.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练集: {len(train_data)} 条 ({len(train_files)} 个对话) → {train_path}")
    print(f"评估集: {len(eval_data)} 条 ({len(eval_files)} 个对话) → {eval_path}")
    print(f"来源文件重叠: {len(overlap)} (应为0)")
    
    # 分布统计
    label_4_names = {0:'积极', 1:'激活消极', 2:'非激活消极', 3:'平静'}
    for name, data in [('训练集', train_data), ('评估集', eval_data)]:
        c = Counter(s['label_4'] for s in data)
        print(f"\n{name} 4类分布:")
        for k in sorted(c):
            print(f"  {label_4_names[k]}: {c[k]} ({c[k]/len(data)*100:.1f}%)")


if __name__ == '__main__':
    print("=" * 60)
    print("HMTL 完整数据集预处理 V2（按对话划分，防数据泄露）")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    file_samples = process_files()
    split_by_file(file_samples)
    
    print("\n✅ 数据预处理V2完成！无数据泄露！")
