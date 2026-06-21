#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将原始对话JSON转为HMTL训练格式
输入: d:\silent like onion\03_数据\文本已标注数据\*.json
输出: d:\bigcreate\05_数据文件\full_training_set_hmtl.json
      d:\bigcreate\05_数据文件\full_eval_set_hmtl.json
"""

import json
import os
import random
from collections import Counter

# ============ 配置 ============
INPUT_DIR = r'd:\silent like onion\03_数据\文本已标注数据'
OUTPUT_DIR = r'd:\bigcreate\05_数据文件'
TRAIN_RATIO = 0.85  # 85%训练，15%评估
RANDOM_SEED = 42

# ============ 标签映射 ============

# 16类原始标签 → 7类
EMOTION_7_MAP = {
    '愤怒': 0, '焦虑': 1, '快乐': 2, '悲伤': 3,
    '失望': 4, '支持': 5, '平静': 6,
    # 细粒度映射
    '生气': 0,
    '紧张': 1, '担心': 1, '害怕': 1, '恐惧': 1, '困惑': 1, '犹豫': 1,
    '兴奋': 2, '激动': 2, '希望': 2, '期待': 2, '自信': 2,
    '沮丧': 3, '无助': 3,
    '理解': 5, '安慰': 5, '鼓励': 5,
    '放松': 6, '坚定': 6,
    # 额外
    '感激': 2,  # 感激→快乐
    '果断': 6,  # 果断→平静
}

# 7类 → 4类
LABEL_7_TO_4 = {
    0: 1,  # 愤怒 → 激活消极
    1: 1,  # 焦虑 → 激活消极
    2: 0,  # 快乐 → 积极
    3: 2,  # 悲伤 → 非激活消极
    4: 2,  # 失望 → 非激活消极
    5: 0,  # 支持 → 积极
    6: 3,  # 平静 → 平静
}

# 4类 → 3类极性
LABEL_4_TO_3 = {
    0: 0,  # 积极 → 正面
    1: 1,  # 激活消极 → 负面
    2: 1,  # 非激活消极 → 负面
    3: 2,  # 平静 → 中性
}

# Arousal/Valence 基准值（按7类）
AROUSAL_MAP = {
    0: 0.85,  # 愤怒 - 高唤醒
    1: 0.75,  # 焦虑 - 较高唤醒
    2: 0.70,  # 快乐 - 中高唤醒
    3: 0.30,  # 悲伤 - 低唤醒
    4: 0.35,  # 失望 - 低唤醒
    5: 0.50,  # 支持 - 中等唤醒
    6: 0.25,  # 平静 - 低唤醒
}

VALENCE_MAP = {
    0: -0.80,  # 愤怒 - 强负效价
    1: -0.60,  # 焦虑 - 负效价
    2:  0.75,  # 快乐 - 正效价
    3: -0.70,  # 悲伤 - 负效价
    4: -0.50,  # 失望 - 负效价
    5:  0.60,  # 支持 - 正效价
    6:  0.05,  # 平静 - 中性
}


def process_files():
    """处理所有对话文件，提取句子级样本"""
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    all_samples = []
    skipped = 0
    label_counter = Counter()
    
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
        
        for msg in data:
            if msg.get('role') != 'client' or 'annotation' not in msg:
                continue
            
            for ann in msg['annotation']:
                sentence = ann.get('sentence', '').strip()
                label_name = ann.get('label', '').strip()
                
                if not sentence or not label_name:
                    continue
                if len(sentence) < 2:
                    continue
                if label_name not in EMOTION_7_MAP:
                    skipped += 1
                    continue
                
                label_7 = EMOTION_7_MAP[label_name]
                label_4 = LABEL_7_TO_4[label_7]
                label_3 = LABEL_4_TO_3[label_4]
                
                # 加一点随机扰动到arousal/valence
                noise_a = random.uniform(-0.08, 0.08)
                noise_v = random.uniform(-0.08, 0.08)
                arousal = max(0.0, min(1.0, AROUSAL_MAP[label_7] + noise_a))
                valence = max(-1.0, min(1.0, VALENCE_MAP[label_7] + noise_v))
                
                sample = {
                    'text': sentence,
                    'original_emotion': label_name,
                    'label_7': label_7,
                    'label_4': label_4,
                    'label_3': label_3,
                    'arousal': round(arousal, 4),
                    'valence': round(valence, 4),
                    'source': fname,
                    'id': f"{fname}_{len(all_samples)}"
                }
                all_samples.append(sample)
                label_counter[label_name] += 1
    
    print(f"总样本数: {len(all_samples)}")
    print(f"跳过: {skipped}")
    print(f"\n标签分布:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")
    
    return all_samples


def split_and_save(samples):
    """按比例划分训练集和评估集并保存"""
    random.seed(RANDOM_SEED)
    random.shuffle(samples)
    
    split_idx = int(len(samples) * TRAIN_RATIO)
    train_data = samples[:split_idx]
    eval_data = samples[split_idx:]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, 'full_training_set_hmtl.json')
    eval_path = os.path.join(OUTPUT_DIR, 'full_eval_set_hmtl.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练集: {len(train_data)} 条 → {train_path}")
    print(f"评估集: {len(eval_data)} 条 → {eval_path}")
    
    # 统计4类分布
    train_4 = Counter(s['label_4'] for s in train_data)
    eval_4 = Counter(s['label_4'] for s in eval_data)
    label_4_names = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}
    
    print(f"\n训练集4类分布:")
    for k in sorted(train_4):
        print(f"  {label_4_names[k]}: {train_4[k]} ({train_4[k]/len(train_data)*100:.1f}%)")
    
    print(f"\n评估集4类分布:")
    for k in sorted(eval_4):
        print(f"  {label_4_names[k]}: {eval_4[k]} ({eval_4[k]/len(eval_data)*100:.1f}%)")


if __name__ == '__main__':
    print("=" * 60)
    print("HMTL 完整数据集预处理")
    print("=" * 60)
    
    samples = process_files()
    split_and_save(samples)
    
    print("\n✅ 数据预处理完成！")
