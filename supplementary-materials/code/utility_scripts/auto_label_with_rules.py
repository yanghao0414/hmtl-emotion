#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI辅助标注工具 - 基于规则的预标注
使用口诀和关键词进行初步标注，标注置信度，供人工复核
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 情绪关键词库（基于口诀）
EMOTION_KEYWORDS = {
    "愤怒": {
        "strong": ["愤怒", "生气", "气死", "火大", "暴怒"],
        "medium": ["气人", "讨厌", "烦人", "可恶", "过分"],
        "weak": ["不爽", "烦", "不满"],
    },
    "快乐": {
        "strong": ["开心", "高兴", "快乐", "兴奋", "激动"],
        "medium": ["太好了", "真棒", "不错", "满意", "欣慰"],
        "weak": ["好", "嗯"],
    },
    "焦虑": {
        "strong": ["害怕", "恐惧", "担心", "紧张", "焦虑"],
        "medium": ["不安", "忧虑", "犹豫", "困惑"],
        "weak": ["会不会", "万一", "如果"],
        "pattern": [r"会不会.*", r"万一.*", r".*怎么办"],
    },
    "悲伤": {
        "strong": ["难过", "痛苦", "伤心", "想哭", "心痛"],
        "medium": ["失落", "沮丧", "无助", "绝望"],
        "weak": ["唉", "哎"],
    },
    "失望": {
        "strong": ["失望", "太失望"],
        "medium": ["没想到", "原来如此", "算了"],
        "weak": ["哦"],
    },
    "支持": {
        "strong": ["理解", "支持", "加油", "陪伴", "拥抱"],
        "medium": ["鼓励", "安慰", "帮助", "关心"],
        "weak": ["可以", "会"],
        "pattern": [r"我.*理解.*你", r"我.*支持.*你", r".*加油.*"],
    },
    "平静": {
        "strong": ["实际上", "根据", "一般来说", "通常"],
        "medium": ["可以", "建议", "方法", "步骤"],
        "weak": ["是", "的"],
        "pattern": [r"^实际上.*", r"^根据.*", r"^你可以.*"],
    },
}

# 极性映射
POLARITY_MAP = {
    "快乐": "积极",
    "支持": "积极",
    "平静": "平静",
    "焦虑": "消极",
    "悲伤": "消极",
    "失望": "消极",
    "愤怒": "消极",
}


def calculate_emotion_score(text: str, emotion: str) -> Tuple[float, List[str]]:
    """
    计算文本对某个情绪的匹配分数
    返回: (分数, 匹配到的关键词列表)
    """
    score = 0.0
    matched = []
    keywords = EMOTION_KEYWORDS.get(emotion, {})
    
    # 强关键词: +1.0
    for kw in keywords.get("strong", []):
        if kw in text:
            score += 1.0
            matched.append(f"{kw}(强)")
    
    # 中关键词: +0.5
    for kw in keywords.get("medium", []):
        if kw in text:
            score += 0.5
            matched.append(f"{kw}(中)")
    
    # 弱关键词: +0.2
    for kw in keywords.get("weak", []):
        if kw in text:
            score += 0.2
            matched.append(f"{kw}(弱)")
    
    # 正则模式: +0.8
    for pattern in keywords.get("pattern", []):
        if re.search(pattern, text):
            score += 0.8
            matched.append(f"[模式:{pattern}]")
    
    return score, matched


def predict_emotion_with_confidence(text: str) -> Dict:
    """
    预测文本的情绪及置信度
    返回: {
        "emotion": "情绪",
        "polarity": "极性",
        "confidence": 0.0-1.0,
        "scores": {"各情绪": 分数},
        "matched_keywords": ["匹配的关键词"]
    }
    """
    # 计算所有情绪的分数
    all_scores = {}
    all_matches = {}
    
    for emotion in EMOTION_KEYWORDS.keys():
        score, matched = calculate_emotion_score(text, emotion)
        all_scores[emotion] = score
        all_matches[emotion] = matched
    
    # 找出最高分
    if not all_scores or max(all_scores.values()) == 0:
        return {
            "emotion": "平静",
            "polarity": "平静",
            "confidence": 0.1,
            "scores": all_scores,
            "matched_keywords": [],
            "note": "未匹配任何关键词，默认为平静"
        }
    
    # 获取最高分情绪
    top_emotion = max(all_scores.items(), key=lambda x: x[1])
    emotion = top_emotion[0]
    score = top_emotion[1]
    
    # 计算置信度 (0-1之间)
    # 如果最高分远超其他分数，置信度高
    sorted_scores = sorted(all_scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        confidence = min(0.9, score / 3.0 + gap / 2.0)
    else:
        confidence = min(0.9, score / 2.0)
    
    return {
        "emotion": emotion,
        "polarity": POLARITY_MAP.get(emotion, "平静"),
        "confidence": round(confidence, 2),
        "scores": all_scores,
        "matched_keywords": all_matches[emotion],
    }


def auto_label_samples(
    input_texts: List[str],
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    批量预标注样本
    只返回置信度 >= threshold 的样本
    """
    results = []
    
    for text in input_texts:
        if not text or len(text.strip()) < 3:
            continue
        
        pred = predict_emotion_with_confidence(text)
        
        if pred["confidence"] >= confidence_threshold:
            results.append({
                "text": text,
                "main_emotion": pred["emotion"],
                "polarity": pred["polarity"],
                "confidence": pred["confidence"],
                "matched_keywords": pred["matched_keywords"],
                "scores": pred["scores"],
                "source": "auto_labeled",
                "need_review": pred["confidence"] < 0.7,  # 低于0.7需要人工复核
            })
    
    return results


def filter_by_emotion(
    samples: List[Dict],
    target_emotion: str,
    min_confidence: float = 0.6,
    max_samples: int = 100
) -> List[Dict]:
    """
    筛选特定情绪的高质量样本
    用于补充少数类
    """
    filtered = [
        s for s in samples
        if s["main_emotion"] == target_emotion
        and s["confidence"] >= min_confidence
    ]
    
    # 按置信度排序
    filtered.sort(key=lambda x: x["confidence"], reverse=True)
    
    return filtered[:max_samples]


def main():
    """主函数：从困难样本中筛选愤怒和快乐"""
    print("="*60)
    print("AI辅助标注工具 - 预标注愤怒和快乐样本")
    print("="*60)
    
    # 加载困难样本
    print("\n[1/5] 加载困难样本...")
    try:
        with open(r"d:\silent like onion\hard_samples.json", "r", encoding="utf-8") as f:
            hard_samples = json.load(f)
        
        # 提取文本
        if isinstance(hard_samples, list):
            texts = [s.get("text", s) if isinstance(s, dict) else s for s in hard_samples]
        else:
            texts = list(hard_samples.values()) if isinstance(hard_samples, dict) else []
        
        print(f"✓ 已加载 {len(texts)} 条困难样本")
    except FileNotFoundError:
        print("✗ 未找到 hard_samples.json，使用评估集作为示例")
        with open(str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json"), "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            texts = [item["text"] for item in eval_data][:1000]
        print(f"✓ 使用评估集的 {len(texts)} 条样本")
    
    # 预标注
    print("\n[2/5] AI预标注中...")
    labeled = auto_label_samples(texts, confidence_threshold=0.5)
    print(f"✓ 预标注完成，高置信度样本: {len(labeled)} 条")
    
    # 统计分布
    print("\n[3/5] 预标注结果分布:")
    emotion_dist = Counter([s["main_emotion"] for s in labeled])
    for emotion, count in emotion_dist.most_common():
        print(f"  {emotion}: {count}条")
    
    # 筛选愤怒和快乐
    print("\n[4/5] 筛选目标情绪样本...")
    anger_samples = filter_by_emotion(labeled, "愤怒", min_confidence=0.6, max_samples=100)
    joy_samples = filter_by_emotion(labeled, "快乐", min_confidence=0.6, max_samples=100)
    
    print(f"  愤怒: {len(anger_samples)}条 (置信度 >= 0.6)")
    print(f"  快乐: {len(joy_samples)}条 (置信度 >= 0.6)")
    
    # 保存结果
    print("\n[5/5] 保存预标注结果...")
    
    output = {
        "summary": {
            "total_labeled": len(labeled),
            "anger_candidates": len(anger_samples),
            "joy_candidates": len(joy_samples),
            "need_review": sum(1 for s in labeled if s["need_review"]),
        },
        "anger_samples": anger_samples[:80],  # 保存前80条
        "joy_samples": joy_samples[:80],      # 保存前80条
        "all_labeled": labeled,
    }
    
    with open(str(_PROJECT_ROOT / "05_数据文件" / "auto_labeled_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("✓ 已保存到: auto_labeled_candidates.json")
    
    # 显示示例
    print("\n" + "="*60)
    print("示例：愤怒候选（前3条）")
    print("="*60)
    for i, sample in enumerate(anger_samples[:3], 1):
        print(f"\n{i}. 文本: {sample['text'][:50]}...")
        print(f"   情绪: {sample['main_emotion']} | 置信度: {sample['confidence']}")
        print(f"   匹配: {', '.join(sample['matched_keywords'])}")
    
    print("\n" + "="*60)
    print("示例：快乐候选（前3条）")
    print("="*60)
    for i, sample in enumerate(joy_samples[:3], 1):
        print(f"\n{i}. 文本: {sample['text'][:50]}...")
        print(f"   情绪: {sample['main_emotion']} | 置信度: {sample['confidence']}")
        print(f"   匹配: {', '.join(sample['matched_keywords'])}")
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)
    print("\n下一步：")
    print("1. 打开 auto_labeled_candidates.json")
    print("2. 复核置信度 >= 0.6 的样本")
    print("3. 修改不准确的标注")
    print("4. 添加到训练集")
    print("\n预计复核时间: 1-2小时（比纯手工快5倍）")


if __name__ == "__main__":
    main()
