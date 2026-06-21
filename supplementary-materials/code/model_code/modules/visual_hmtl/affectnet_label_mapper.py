#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AffectNet 标签映射器
将 AffectNet 的 8 类情绪映射到 HMTL 的 7 类情绪
并生成 A/V 连续值标签

映射规则 2024-12-08
- anger → 愤怒
- fear → 焦虑  (恐惧映射为焦虑)
- happy → 快乐
- surprise → 快乐
- sad → 悲伤
- disgust → 悲伤/失望
- contempt → 失望
- neutral → 平静
"""

import os
import pandas as pd
from typing import Dict, Tuple

# ============================================================
# AffectNet 8类 → HMTL 7类 映射表
# 包含 A/V 基础值，基于 Russell 情感环形模型
# ============================================================
AFFECTNET_TO_HMTL = {
    'anger': {
        'label_7': 0,        # 愤怒
        'label_4': 1,        # 激活消极
        'label_3': 1,        # 消极
        'base_arousal': 0.9,
        'base_valence': -0.8
    },
    'fear': {
        'label_7': 1,        # 焦虑
        'label_4': 1,        # 激活消极
        'label_3': 1,        # 消极
        'base_arousal': 0.8,
        'base_valence': -0.6
    },
    'happy': {
        'label_7': 2,        # 快乐
        'label_4': 0,        # 积极
        'label_3': 0,        # 积极
        'base_arousal': 0.7,
        'base_valence': 0.9
    },
    'surprise': {
        'label_7': 2,        # 快乐
        'label_4': 0,        # 积极
        'label_3': 0,        # 积极
        'base_arousal': 0.8,
        'base_valence': 0.7
    },
    'sad': {
        'label_7': 3,        # 悲伤
        'label_4': 2,        # 非激活消极
        'label_3': 1,        # 消极
        'base_arousal': 0.3,
        'base_valence': -0.7
    },
    'disgust': {
        'label_7': 3,        # 悲伤/失望
        'label_4': 2,        # 非激活消极
        'label_3': 1,        # 消极
        'base_arousal': 0.4,
        'base_valence': -0.6
    },
    'contempt': {
        'label_7': 4,        # 失望
        'label_4': 2,        # 非激活消极
        'label_3': 1,        # 消极
        'base_arousal': 0.4,
        'base_valence': -0.4
    },
    'neutral': {
        'label_7': 6,        # 平静
        'label_4': 3,        # 平静
        'label_3': 2,        # 平静
        'base_arousal': 0.2,
        'base_valence': 0.1
    }
}

# 7类情绪名称
LABEL_7_NAMES = {
    0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤',
    4: '失望', 5: '支持', 6: '平静'
}

# 4类情绪名称
LABEL_4_NAMES = {
    0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'
}

# 3类极性名称
LABEL_3_NAMES = {
    0: '积极', 1: '消极', 2: '平静'
}


def map_affectnet_label(original_label: str, confidence: float) -> Dict:
    """
    将 AffectNet 标签映射为 HMTL 标签
    并根据置信度调整 A/V 值
    
    Args:
        original_label: AffectNet 原始标签 (anger, fear, happy, etc.)
        confidence: 置信度 (0-1)
    
    Returns:
        dict: {
            'label_7': int,
            'label_4': int,
            'label_3': int,
            'arousal': float,  # 调整后的唤醒度
            'valence': float,  # 调整后的效价
            'original_label': str,
            'confidence': float
        }
    """
    original_label = original_label.lower().strip()
    
    if original_label not in AFFECTNET_TO_HMTL:
        print(f"[WARNING] 未知标签: {original_label}, 默认使用 neutral")
        original_label = 'neutral'
    
    mapping = AFFECTNET_TO_HMTL[original_label]
    
    # 根据置信度调整 A/V 值
    # 高置信度 → 更接近基础值
    # 低置信度 → 更接近中性值
    adjusted_arousal = mapping['base_arousal'] * confidence + 0.2 * (1 - confidence)
    adjusted_valence = mapping['base_valence'] * confidence + 0.0 * (1 - confidence)
    
    # 确保 arousal 在 [0, 1] 范围内
    adjusted_arousal = max(0.0, min(1.0, adjusted_arousal))
    # 确保 valence 在 [-1, 1] 范围内
    adjusted_valence = max(-1.0, min(1.0, adjusted_valence))
    
    return {
        'label_7': mapping['label_7'],
        'label_4': mapping['label_4'],
        'label_3': mapping['label_3'],
        'arousal': round(adjusted_arousal, 4),
        'valence': round(adjusted_valence, 4),
        'original_label': original_label,
        'confidence': confidence
    }


def process_affectnet_dataset(
    data_dir: str,
    labels_csv: str,
    output_csv: str
) -> pd.DataFrame:
    """
    处理 AffectNet 数据集生成 HMTL 标签
    
    Args:
        data_dir: 数据集根目录 (包含 Train/Test 子目录)
        labels_csv: 原始 labels.csv 路径
        output_csv: 输出 HMTL 标签文件路径
    
    Returns:
        pd.DataFrame: 处理后的数据帧
    """
    print(f"加载标签文件: {labels_csv}")
    df = pd.read_csv(labels_csv)
    
    print(f"总样本数: {len(df)}")
    print(f"原始分布:\n{df['label'].value_counts()}")
    
    # 逐条处理
    results = []
    for idx, row in df.iterrows():
        img_path = row['pth']
        original_label = row['label']
        confidence = float(row['relFCs'])
        
        # 查找图片路径
        full_path = os.path.join(data_dir, 'Train', img_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(data_dir, 'Test', img_path)
        
        # 映射标签
        mapped = map_affectnet_label(original_label, confidence)
        
        results.append({
            'image_path': full_path,
            'original_path': img_path,
            'original_label': original_label,
            'confidence': confidence,
            'label_7': mapped['label_7'],
            'label_4': mapped['label_4'],
            'label_3': mapped['label_3'],
            'arousal': mapped['arousal'],
            'valence': mapped['valence']
        })
        
        if (idx + 1) % 5000 == 0:
            print(f"已处理: {idx + 1}/{len(df)}")
    
    result_df = pd.DataFrame(results)
    
    # 统计分布
    print("\n" + "="*60)
    print("标签分布统计 (7类):")
    print("="*60)
    for label_id in sorted(result_df['label_7'].unique()):
        count = len(result_df[result_df['label_7'] == label_id])
        name = LABEL_7_NAMES.get(label_id, '未知')
        print(f"  [{label_id}] {name}: {count} ({count/len(result_df)*100:.1f}%)")
    
    print("\n标签分布统计 (4类):")
    for label_id in sorted(result_df['label_4'].unique()):
        count = len(result_df[result_df['label_4'] == label_id])
        name = LABEL_4_NAMES.get(label_id, '未知')
        print(f"  [{label_id}] {name}: {count} ({count/len(result_df)*100:.1f}%)")
    
    print(f"\nArousal 范围: [{result_df['arousal'].min():.2f}, {result_df['arousal'].max():.2f}]")
    print(f"Valence 范围: [{result_df['valence'].min():.2f}, {result_df['valence'].max():.2f}]")
    
    # 保存结果
    result_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\n已保存到: {output_csv}")
    
    return result_df


if __name__ == "__main__":
    # 测试映射
    print("="*60)
    print("AffectNet → HMTL 标签映射测试")
    print("="*60)
    
    test_cases = [
        ('happy', 0.95),
        ('happy', 0.60),
        ('anger', 0.90),
        ('anger', 0.55),
        ('contempt', 0.85),
        ('contempt', 0.55),
        ('disgust', 0.90),
        ('disgust', 0.55),
        ('neutral', 0.80),
    ]
    
    print(f"\n{'原始标签':<12} {'置信度':<8} {'→':<3} {'7类':<6} {'A':<6} {'V':<6}")
    print("-"*50)
    
    for label, conf in test_cases:
        result = map_affectnet_label(label, conf)
        emotion_name = LABEL_7_NAMES[result['label_7']]
        print(f"{label:<12} {conf:<8.2f} → {emotion_name:<6} {result['arousal']:<6.2f} {result['valence']:<+6.2f}")
