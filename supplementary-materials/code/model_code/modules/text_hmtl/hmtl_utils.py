#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL工具函数
情绪标签映射和转换
"""

from typing import Dict

# 7类情绪到HMTL多任务标签的映射，基于Russell情感环形模型
EMOTION_MAP = {
    # 7类基本情绪
    '愤怒': {
        'label_4': 1,        # 4类: 激活消极
        'label_3': 1,        # 3类: 消极
        'arousal': 0.9,      # 高激活: 愤怒是高唤醒情绪
        'valence': -0.8      # 负效价: 愤怒是负面情绪
    },
    '焦虑': {
        'label_4': 1,        # 激活消极
        'label_3': 1,        # 消极
        'arousal': 0.7,
        'valence': -0.6
    },
    '快乐': {
        'label_4': 0,        # 4类: 积极
        'label_3': 0,        # 3类: 积极
        'arousal': 0.7,
        'valence': 0.9
    },
    '支持': {
        'label_4': 0,        # 积极
        'label_3': 0,        # 积极
        'arousal': 0.5,
        'valence': 0.7
    },
    '悲伤': {
        'label_4': 2,        # 4类: 非激活消极
        'label_3': 1,        # 3类: 消极
        'arousal': 0.3,
        'valence': -0.7
    },
    '失望': {
        'label_4': 2,        # 非激活消极
        'label_3': 1,        # 消极
        'arousal': 0.4,
        'valence': -0.5
    },
    '平静': {
        'label_4': 3,        # 4类: 平静
        'label_3': 2,        # 3类: 平静
        'arousal': 0.2,
        'valence': 0.3
    },
    
    # 同义词扩展映射
    '生气': {  # 愤怒
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.9,
        'valence': -0.8
    },
    '难过': {  # 悲伤
        'label_4': 2,
        'label_3': 1,
        'arousal': 0.3,
        'valence': -0.6
    },
    '伤心': {  # 悲伤
        'label_4': 2,
        'label_3': 1,
        'arousal': 0.2,
        'valence': -0.7
    },
    '担心': {  # 焦虑
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.7,
        'valence': -0.5
    },
    '紧张': {  # 焦虑
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.6,
        'valence': -0.5
    },
    '烦躁': {  # 焦虑
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.8,
        'valence': -0.6
    },
    '愤怒': {  # 愤怒
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.9,
        'valence': -0.7
    },
    '没办法': {  # 失望
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.5,
        'valence': -0.4
    },
    '无奈': {  # 失望
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.4,
        'valence': -0.3
    },
    '开心': {  # 快乐
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.9,
        'valence': 0.8
    },
    '高兴': {  # 快乐
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.9,
        'valence': 0.8
    },
    '欣慰': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.7
    },
    '感动': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.7
    },
    '满足': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.5,
        'valence': 0.8
    },
    '感恩': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.4,
        'valence': 0.7
    },
    '幸福': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.4,
        'valence': 0.7
    },
    '鼓励': {  # 支持
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.8
    },
    '淡定': {  # 平静
        'label_4': 3,
        'label_3': 2,
        'arousal': 0.2,
        'valence': 0.5
    }
}

# 标签名称映射
LABEL_4_NAMES = {
    0: '积极',
    1: '激活消极',
    2: '非激活消极',
    3: '平静'
}

LABEL_3_NAMES = {
    0: '积极',
    1: '消极',
    2: '平静'
}

# 根据label_4和arousal/valence反推原始情绪
def predict_original_emotion(label_4: int, arousal: float, valence: float) -> str:
    """
    根据HMTL预测结果反推原始7类情绪
    
    Args:
        label_4: 4类情绪标签 (0-3)
        arousal: 唤醒度 (0-1)
        valence: 效价 (-1 to 1)
    
    Returns:
        原始情绪名称
    """
    if label_4 == 0:  # 积极
        if arousal > 0.6:
            return '快乐'
        else:
            return '支持'
    
    elif label_4 == 1:  # 激活消极
        if arousal > 0.8:
            return '愤怒'
        else:
            return '焦虑'
    
    elif label_4 == 2:  # 非激活消极
        if valence < -0.6:
            return '悲伤'
        else:
            return '失望'
    
    else:  # label_4 == 3, 平静
        return '平静'


def get_hmtl_labels(original_label: str) -> Dict:
    """
    获取HMTL多任务标签
    
    Args:
        original_label: 原始7类情绪标签
    
    Returns:
        包含4个任务标签的字典
    """
    if original_label not in EMOTION_MAP:
        print(f"[WARNING] 未知情绪标签: {original_label}, 使用'平静'默认值")
        return EMOTION_MAP['平静']
    
    return EMOTION_MAP[original_label]


def print_emotion_distribution(label_4_counts: Dict, label_3_counts: Dict):
    """打印情绪分布统计"""
    print("\n" + "="*60)
    print("情绪分布统计")
    print("="*60)
    
    print("\n4类分布:")
    total_4 = sum(label_4_counts.values())
    for label_id in sorted(label_4_counts.keys()):
        count = label_4_counts[label_id]
        name = LABEL_4_NAMES.get(label_id, f"未知{label_id}")
        pct = count / total_4 * 100 if total_4 > 0 else 0
        print(f"  [{label_id}] {name}: {count} ({pct:.1f}%)")
    
    print("\n3类分布:")
    total_3 = sum(label_3_counts.values())
    for label_id in sorted(label_3_counts.keys()):
        count = label_3_counts[label_id]
        name = LABEL_3_NAMES.get(label_id, f"未知{label_id}")
        pct = count / total_3 * 100 if total_3 > 0 else 0
        print(f"  [{label_id}] {name}: {count} ({pct:.1f}%)")
    
    print("="*60)
