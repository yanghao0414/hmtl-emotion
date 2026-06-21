#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL标签映射器 - 统一情绪标签管理

多模态情绪识别系统的核心标签映射工具
基于 Russell 情感环形模型 (Circumplex Model of Affect)
- Arousal (唤醒度): 连续值 (0-1)
- Valence (效价): 连续值 (-1 to 1)
"""

from typing import Dict, Tuple, Optional

# ============================================================
# 7类情绪定义 (基础情绪类别)
# ============================================================
EMOTION_7_LABELS = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']

EMOTION_7_MAP = {
    '愤怒': 0, '焦虑': 1, '快乐': 2, '悲伤': 3,
    '失望': 4, '支持': 5, '平静': 6
}

EMOTION_7_NAMES = {
    0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤',
    4: '失望', 5: '支持', 6: '平静'
}


# ============================================================
# 情绪映射表 (包含同义词扩展)
# 基于 Russell 情感环形模型
# ============================================================
EMOTION_MAP = {
    # ========== 7类基本情绪 ==========
    '愤怒': {
        'label_7': 0,        # 7类ID
        'label_4': 1,        # 4类: 激活消极
        'label_3': 1,        # 3类: 消极
        'arousal': 0.9,      # 高唤醒: 愤怒是高激活情绪
        'valence': -0.8      # 负效价: 愤怒是负面情绪
    },
    '焦虑': {
        'label_7': 1,
        'label_4': 1,        # 激活消极
        'label_3': 1,        # 消极
        'arousal': 0.7,
        'valence': -0.6
    },
    '快乐': {
        'label_7': 2,
        'label_4': 0,        # 4类: 积极
        'label_3': 0,        # 3类: 积极
        'arousal': 0.7,
        'valence': 0.9
    },
    '悲伤': {
        'label_7': 3,
        'label_4': 2,        # 4类: 非激活消极
        'label_3': 1,        # 3类: 消极
        'arousal': 0.3,
        'valence': -0.7
    },
    '失望': {
        'label_7': 4,
        'label_4': 2,        # 非激活消极
        'label_3': 1,        # 消极
        'arousal': 0.4,
        'valence': -0.5
    },
    '支持': {
        'label_7': 5,
        'label_4': 0,        # 积极
        'label_3': 0,        # 积极
        'arousal': 0.5,
        'valence': 0.7
    },
    '平静': {
        'label_7': 6,
        'label_4': 3,        # 4类: 平静
        'label_3': 2,        # 3类: 平静
        'arousal': 0.2,
        'valence': 0.3
    },
    
    # ========== 生气 → 愤怒 ==========
    '生气': {
        'label_7': 0,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.9,
        'valence': -0.8
    },
    
    # ========== 同义词 → 焦虑 ==========
    '担心': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.7,
        'valence': -0.5
    },
    '紧张': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.6,
        'valence': -0.5
    },
    '烦躁': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.8,
        'valence': -0.6
    },
    '愤怒': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.9,
        'valence': -0.7
    },
    '没办法': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.5,
        'valence': -0.4
    },
    '无奈': {
        'label_7': 1,
        'label_4': 1,
        'label_3': 1,
        'arousal': 0.4,
        'valence': -0.3
    },
    
    # ========== 同义词 → 快乐 ==========
    '开心': {
        'label_7': 2,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.9,
        'valence': 0.8
    },
    '高兴': {
        'label_7': 2,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.9,
        'valence': 0.8
    },
    '欣慰': {
        'label_7': 2,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.7
    },
    '感动': {
        'label_7': 2,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.7
    },
    '满足': {
        'label_7': 2,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.5,
        'valence': 0.8
    },
    
    # ========== 同义词 → 悲伤 ==========
    '难过': {
        'label_7': 3,
        'label_4': 2,
        'label_3': 1,
        'arousal': 0.3,
        'valence': -0.6
    },
    '伤心': {
        'label_7': 3,
        'label_4': 2,
        'label_3': 1,
        'arousal': 0.2,
        'valence': -0.7
    },
    
    # ========== 同义词 → 支持 ==========
    '感恩': {
        'label_7': 5,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.4,
        'valence': 0.7
    },
    '幸福': {
        'label_7': 5,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.4,
        'valence': 0.7
    },
    '鼓励': {
        'label_7': 5,
        'label_4': 0,
        'label_3': 0,
        'arousal': 0.6,
        'valence': 0.8
    },
    
    # ========== 同义词 → 平静 ==========
    '淡定': {
        'label_7': 6,
        'label_4': 3,
        'label_3': 2,
        'arousal': 0.2,
        'valence': 0.5
    },
    '平常': {
        'label_7': 6,
        'label_4': 3,
        'label_3': 2,
        'arousal': 0.3,
        'valence': 0.4
    }
}


# ============================================================
# 4类情绪分类
# ============================================================
NUM_4_EMOTION_CLASSES = 4  # 情绪类别数

LABEL_4_NAMES = {
    0: '积极',
    1: '激活消极',
    2: '非激活消极',
    3: '平静'
}

LABEL_4_MAP = {v: k for k, v in LABEL_4_NAMES.items()}


# ============================================================
# 3类情感极性
# ============================================================
NUM_3_POLARITY_CLASSES = 3  # 极性类别数

LABEL_3_NAMES = {
    0: '积极',
    1: '消极',
    2: '平静'
}

LABEL_3_MAP = {v: k for k, v in LABEL_3_NAMES.items()}


# ============================================================
# HMTL标签获取函数
# ============================================================
def get_hmtl_labels(emotion: str) -> Dict:
    """
    获取HMTL多任务标签
    
    Args:
        emotion: 情绪名称，支持7类基本情绪和20+同义词
    
    Returns:
        dict: {
            'label_7': int,      # 7类ID (0-6)
            'label_4': int,      # 4类ID (0-3)
            'label_3': int,      # 3类ID (0-2)
            'arousal': float,    # 唤醒度 (0-1)
            'valence': float     # 效价 (-1 to 1)
        }
    
    Examples:
        >>> get_hmtl_labels('快乐')
        {'label_7': 2, 'label_4': 0, 'label_3': 0, 'arousal': 0.7, 'valence': 0.9}
        
        >>> get_hmtl_labels('开心')  # 同义词映射
        {'label_7': 2, 'label_4': 0, 'label_3': 0, 'arousal': 0.9, 'valence': 0.8}
    """
    if emotion not in EMOTION_MAP:
        print(f"[WARNING] 未知情绪: '{emotion}', 使用'平静'默认值")
        return EMOTION_MAP['平静']
    
    return EMOTION_MAP[emotion]


def predict_emotion_from_av(arousal: float, valence: float, 
                            label_4: Optional[int] = None) -> str:
    """
    根据Arousal/Valence反推7类情绪，基于Russell情感环形模型
    
    Args:
        arousal: 唤醒度 (0-1)
        valence: 效价 (-1 to 1)
        label_4: 可选4类标签约束
    
    Returns:
        str: 7类情绪名称
    
    Russell模型说明:
        - 高唤醒 + 正效价 → 快乐
        - 高唤醒 + 负效价 → 愤怒/焦虑 (激活消极)
        - 低唤醒 + 负效价 → 悲伤/失望 (非激活消极)
        - 低唤醒 + 正效价 → 支持/平静 (积极)
    """
    if label_4 is not None:
        # 根据4类约束
        if label_4 == 0:  # 积极
            return '快乐' if arousal > 0.6 else '支持'
        elif label_4 == 1:  # 激活消极
            return '愤怒' if arousal > 0.8 else '焦虑'
        elif label_4 == 2:  # 非激活消极
            return '悲伤' if valence < -0.6 else '失望'
        else:  # label_4 == 3, 平静
            return '平静'
    
    # 纯Arousal/Valence推断
    if arousal > 0.6:  # 高唤醒
        if valence > 0.5:
            return '快乐'
        elif valence < -0.5:
            return '愤怒' if arousal > 0.8 else '焦虑'
        else:
            return '焦虑'
    else:  # 低唤醒
        if valence > 0.5:
            return '支持' if valence > 0.6 else '平静'
        elif valence < -0.5:
            return '悲伤' if valence < -0.6 else '失望'
        else:
            return '平静'


def get_emotion_coordinates() -> Dict[str, Tuple[float, float]]:
    """
    获取所有情绪在Russell环形模型中的坐标
    
    Returns:
        dict: {emotion_name: (arousal, valence), ...}
    
    可用于可视化情绪分布
    """
    coordinates = {}
    for emotion, labels in EMOTION_MAP.items():
        coordinates[emotion] = (labels['arousal'], labels['valence'])
    return coordinates


def print_emotion_stats(label_counts: Dict[str, int]):
    """
    打印情绪分布统计
    
    Args:
        label_counts: 情绪计数字典 {'快乐': 80, '愤怒': 45, ...}
    """
    print("\n" + "="*60)
    print("情绪分布统计")
    print("="*60)
    
    total = sum(label_counts.values())
    
    # 7类分布
    emotion_7_counts = {emo: 0 for emo in EMOTION_7_LABELS}
    for emotion, count in label_counts.items():
        labels = get_hmtl_labels(emotion)
        main_emotion = EMOTION_7_NAMES[labels['label_7']]
        emotion_7_counts[main_emotion] += count
    
    print("\n7类分布:")
    for emotion in EMOTION_7_LABELS:
        count = emotion_7_counts[emotion]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {emotion}: {count} ({pct:.1f}%)")
    
    # 4类分布
    label_4_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for emotion, count in label_counts.items():
        labels = get_hmtl_labels(emotion)
        label_4_counts[labels['label_4']] += count
    
    print("\n4类分布:")
    for label_id, name in LABEL_4_NAMES.items():
        count = label_4_counts[label_id]
        pct = count / total * 100 if total > 0 else 0
        print(f"  [{label_id}] {name}: {count} ({pct:.1f}%)")
    
    # 3类分布
    label_3_counts = {0: 0, 1: 0, 2: 0}
    for emotion, count in label_counts.items():
        labels = get_hmtl_labels(emotion)
        label_3_counts[labels['label_3']] += count
    
    print("\n3类分布:")
    for label_id, name in LABEL_3_NAMES.items():
        count = label_3_counts[label_id]
        pct = count / total * 100 if total > 0 else 0
        print(f"  [{label_id}] {name}: {count} ({pct:.1f}%)")
    
    print("="*60)


# ============================================================
# 多模态融合预测
# ============================================================
def fuse_predictions(text_probs: Dict[str, float],
                     audio_probs: Optional[Dict[str, float]] = None,
                     video_probs: Optional[Dict[str, float]] = None,
                     weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> str:
    """
    文本+音频+视觉多模态融合预测
    
    Args:
        text_probs: 文本模态的7类概率
        audio_probs: 音频模态的7类概率
        video_probs: 视觉模态的7类概率
        weights: 模态权重 (text, audio, video)
    
    Returns:
        str: 融合后的情绪名称
    
    Example:
        >>> fuse_predictions(
        ...     text_probs={'快乐': 0.7, '支持': 0.2, '平静': 0.1},
        ...     audio_probs={'快乐': 0.6, '支持': 0.3, '平静': 0.1}
        ... )
        '快乐'
    """
    # 初始化融合概率
    fused_probs = {emo: 0.0 for emo in EMOTION_7_LABELS}
    
    # 加权融合
    w_text, w_audio, w_video = weights
    
    for emotion in EMOTION_7_LABELS:
        if emotion in text_probs:
            fused_probs[emotion] += w_text * text_probs[emotion]
        
        if audio_probs and emotion in audio_probs:
            fused_probs[emotion] += w_audio * audio_probs[emotion]
        
        if video_probs and emotion in video_probs:
            fused_probs[emotion] += w_video * video_probs[emotion]
    
    # 返回概率最高的情绪
    return max(fused_probs.items(), key=lambda x: x[1])[0]


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("HMTL 标签映射器测试")
    print("="*60)
    
    # 测试基本映射
    print("\n测试1: 7类基本情绪映射:")
    for emotion in EMOTION_7_LABELS:
        labels = get_hmtl_labels(emotion)
        print(f"{emotion:4s}: 7={labels['label_7']}, 4={labels['label_4']}, "
              f"3={labels['label_3']}, A={labels['arousal']:.1f}, V={labels['valence']:+.1f}")
    
    # 测试同义词映射
    print("\n测试2: 同义词映射:")
    fine_emotions = ['开心', '担心', '难过', '淡定']
    for emotion in fine_emotions:
        labels = get_hmtl_labels(emotion)
        main = EMOTION_7_NAMES[labels['label_7']]
        print(f"{emotion:4s} → {main}")
    
    # 测试A/V反推
    print("\n测试3: A/V反推情绪:")
    test_cases = [
        (0.9, 0.8, "预期: 快乐"),
        (0.9, -0.8, "预期: 愤怒"),
        (0.3, -0.7, "预期: 悲伤"),
        (0.5, 0.7, "预期: 支持"),
    ]
    for arousal, valence, expected in test_cases:
        pred = predict_emotion_from_av(arousal, valence)
        print(f"A={arousal:.1f}, V={valence:+.1f} → {pred} ({expected})")
    
    print("\n" + "="*60)
    print("测试完成")
