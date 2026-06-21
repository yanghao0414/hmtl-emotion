#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Audio HMTL 生成 7类标签
从现有的 4类 + A/V 推断 7类情绪
"""

import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from label_mapper import predict_emotion_from_av, EMOTION_7_MAP

# 7类情绪映射
EMOTION_7_NAMES = {
    0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤',
    4: '失望', 5: '支持', 6: '平静'
}


def infer_label_7(label_4: int, arousal: float, valence: float) -> int:
    """
    从 4类标签 + A/V 推断 7类标签
    
    4类定义:
        0: 积极
        1: 激活型消极
        2: 非激活型消极
        3: 平静
    
    7类定义:
        0: 愤怒 (激活型消极, 高arousal)
        1: 焦虑 (激活型消极, 中arousal)
        2: 快乐 (积极, 高arousal)
        3: 悲伤 (非激活型消极, 低valence)
        4: 失望 (非激活型消极, 中valence)
        5: 支持 (积极, 低arousal)
        6: 平静
    """
    if label_4 == 0:  # 积极
        if arousal > 0.5:
            return 2  # 快乐
        else:
            return 5  # 支持
    
    elif label_4 == 1:  # 激活型消极
        if arousal > 0.7:
            return 0  # 愤怒
        else:
            return 1  # 焦虑
    
    elif label_4 == 2:  # 非激活型消极
        if valence < -0.5:
            return 3  # 悲伤
        else:
            return 4  # 失望
    
    else:  # label_4 == 3, 平静
        return 6  # 平静


def generate_label7_csv(input_csv: str, output_csv: str):
    """
    读取现有标签文件，添加 label_7 列
    """
    print(f"读取: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"原始样本数: {len(df)}")
    print(f"原始列: {list(df.columns)}")
    
    # 推断 7类标签
    label_7_list = []
    for idx, row in df.iterrows():
        label_4 = int(row['label_4_emotion'])
        arousal = float(row['true_arousal'])
        valence = float(row['true_valence'])
        
        label_7 = infer_label_7(label_4, arousal, valence)
        label_7_list.append(label_7)
    
    df['label_7_emotion'] = label_7_list
    
    # 统计分布
    print("\n7类标签分布:")
    for label_id in sorted(df['label_7_emotion'].unique()):
        count = len(df[df['label_7_emotion'] == label_id])
        name = EMOTION_7_NAMES.get(label_id, '未知')
        print(f"  [{label_id}] {name}: {count} ({count/len(df)*100:.1f}%)")
    
    # 保存
    df.to_csv(output_csv, index=False)
    print(f"\n已保存到: {output_csv}")
    
    return df


if __name__ == "__main__":
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parents[3]
    input_csv = str(_PROJECT_ROOT / "05_数据文件" / "audio_hmtl_labels.csv")
    output_csv = str(_PROJECT_ROOT / "05_数据文件" / "audio_hmtl_labels_v2.csv")
    
    generate_label7_csv(input_csv, output_csv)
