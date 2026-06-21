#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将SMP2020-EWECT数据集转换为HMTL训练格式
6类情绪 -> 映射到4类/7类/3类 + arousal/valence
"""

import json
import os
import random
from collections import Counter

INPUT_DIR = r'd:\bigcreate\public_datasets\SMP2020-EWECT\data\raw'
OUTPUT_DIR = r'd:\bigcreate\05_数据文件'
RANDOM_SEED = 42

# SMP2020 6类 -> HMTL 7类映射
# SMP: happy, angry, sad, fear, surprise, neutral
# HMTL 7类: 0愤怒, 1焦虑, 2快乐, 3悲伤, 4失望, 5支持, 6平静
SMP_TO_7 = {
    'angry': 0,      # 愤怒 -> 愤怒
    'fear': 1,        # 恐惧 -> 焦虑
    'happy': 2,       # 积极 -> 快乐
    'sad': 3,         # 悲伤 -> 悲伤
    'surprise': 2,    # 惊奇 -> 快乐（多为正面惊奇）
    'neutral': 6,     # 无情绪 -> 平静
}

LABEL_7_TO_4 = {0:1, 1:1, 2:0, 3:2, 4:2, 5:0, 6:3}
LABEL_4_TO_3 = {0:0, 1:1, 2:1, 3:2}

AROUSAL_MAP = {0:0.85, 1:0.75, 2:0.70, 3:0.30, 4:0.35, 5:0.50, 6:0.25}
VALENCE_MAP = {0:-0.80, 1:-0.60, 2:0.75, 3:-0.70, 4:-0.50, 5:0.60, 6:0.05}

LABEL_4_NAMES = {0:'积极', 1:'激活消极', 2:'非激活消极', 3:'平静'}
LABEL_7_NAMES = {0:'愤怒', 1:'焦虑', 2:'快乐', 3:'悲伤', 4:'失望', 5:'支持', 6:'平静'}


def convert_smp_to_hmtl(input_path, output_path, set_name):
    """转换SMP2020数据为HMTL格式"""
    random.seed(RANDOM_SEED)
    
    raw = json.loads(open(input_path, 'r', encoding='utf-8').read())
    
    samples = []
    skipped = 0
    for item in raw:
        text = item['content'].strip()
        label = item.get('label', '')
        
        if not text or len(text) < 2:
            skipped += 1
            continue
        if label not in SMP_TO_7:
            skipped += 1
            continue
        
        label_7 = SMP_TO_7[label]
        label_4 = LABEL_7_TO_4[label_7]
        label_3 = LABEL_4_TO_3[label_4]
        
        noise_a = random.uniform(-0.08, 0.08)
        noise_v = random.uniform(-0.08, 0.08)
        arousal = max(0.0, min(1.0, AROUSAL_MAP[label_7] + noise_a))
        valence = max(-1.0, min(1.0, VALENCE_MAP[label_7] + noise_v))
        
        samples.append({
            'text': text,
            'original_emotion': label,
            'label_7': label_7,
            'label_4': label_4,
            'label_3': label_3,
            'arousal': round(arousal, 4),
            'valence': round(valence, 4),
            'source': f'smp2020_{set_name}',
            'id': f'smp2020_{item["id"]}',
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n{set_name}: {len(raw)} -> {len(samples)} 条 (跳过{skipped})")
    
    # 分布
    c4 = Counter(s['label_4'] for s in samples)
    c7 = Counter(s['label_7'] for s in samples)
    print(f"  4类分布:")
    for k in sorted(c4):
        print(f"    {LABEL_4_NAMES[k]}: {c4[k]} ({c4[k]/len(samples)*100:.1f}%)")
    print(f"  7类分布:")
    for k in sorted(c7):
        if c7[k] > 0:
            print(f"    {LABEL_7_NAMES[k]}: {c7[k]} ({c7[k]/len(samples)*100:.1f}%)")
    
    return samples


if __name__ == '__main__':
    print("=" * 60)
    print("SMP2020-EWECT -> HMTL格式转换")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 训练集 = usual_train
    train_path = os.path.join(OUTPUT_DIR, 'smp2020_training_set_hmtl.json')
    convert_smp_to_hmtl(
        os.path.join(INPUT_DIR, 'usual_train.txt'),
        train_path, 'train')
    
    # 评估集 = usual_eval + usual_test 合并（共7000条）
    eval_data = []
    for fname, sname in [('usual_eval_labeled.txt', 'eval'), ('usual_test_labeled.txt', 'test')]:
        samples = convert_smp_to_hmtl(
            os.path.join(INPUT_DIR, fname),
            os.path.join(OUTPUT_DIR, f'smp2020_{sname}_hmtl.json'),
            sname)
        eval_data.extend(samples)
    
    # 合并eval+test作为完整评估集
    eval_combined_path = os.path.join(OUTPUT_DIR, 'smp2020_eval_set_hmtl.json')
    with open(eval_combined_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n合并评估集: {len(eval_data)} 条 -> {eval_combined_path}")
    
    c4 = Counter(s['label_4'] for s in eval_data)
    print(f"  4类分布:")
    for k in sorted(c4):
        print(f"    {LABEL_4_NAMES[k]}: {c4[k]} ({c4[k]/len(eval_data)*100:.1f}%)")
    
    print(f"\n✅ SMP2020数据转换完成！")
    print(f"训练集: {train_path}")
    print(f"评估集: {eval_combined_path}")
