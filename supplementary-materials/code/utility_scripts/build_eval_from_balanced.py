#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 03_数据/balanced_training_samples.json 生成评估集 eval_set.json

用法：
    python 08_工具脚本\build_eval_from_balanced.py

说明：
- 原始文件 balanced_training_samples.json 的结构是：
    {
      "samples": [文本1, 文本2, ...],
      "labels":  [标签1, 标签2, ...]
    }
- 本脚本会将其转换成 eval_set.json，结构为：
    [
      {
        "text": "...",
        "main_emotion": "...",
        "polarity": "积极" / "消极",
        "source": "balanced_training_samples"
      },
      ...
    ]
- 后续所有模型版本可以统一在 eval_set.json 上做评估比较。
"""

import json
import os
from typing import List, Dict

BALANCED_PATH = r"d:\\silent like onion\\03_数据\\balanced_training_samples.json"
EVAL_PATH = r"d:\\bigcreate\\eval_set.json"

# 简单的情绪 -> 极性映射，根据直觉规则设计，可后续再调整
POSITIVE_LABELS = {
    "理解", "平静", "放松", "坚定", "快乐", "鼓励", "支持", "安慰",
    "希望", "期待", "自信", "兴奋", "激动"
}
NEGATIVE_LABELS = {
    "沮丧", "无助", "紧张", "悲伤", "失望", "困惑", "犹豫",
    "害怕", "恐惧", "愤怒", "生气", "绝望"
}


def infer_polarity(label: str) -> str:
    """根据细分类标签粗略推断极性（积极/消极）。"""
    if label in POSITIVE_LABELS:
        return "积极"
    if label in NEGATIVE_LABELS:
        return "消极"
    # 默认按中性偏积极处理，也可以改成 "消极" 或 "中性"
    return "平静"


def load_balanced_samples(path: str) -> Dict[str, List[str]]:
    """读取 balanced_training_samples.json，处理前缀异常等问题。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 balanced_training_samples.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # 文件开头有 "yy{" 这种前缀，先找到第一个 '{' 再做 json 解析
    first_brace = raw.find("{")
    if first_brace > 0:
        raw = raw[first_brace:]

    data = json.loads(raw)

    samples = data.get("samples", [])
    labels = data.get("labels", [])

    if len(samples) != len(labels):
        print(f"[WARNING] samples 与 labels 数量不一致: {len(samples)} vs {len(labels)}")

    return {"samples": samples, "labels": labels}


def build_eval_set() -> None:
    balanced = load_balanced_samples(BALANCED_PATH)
    samples: List[str] = balanced["samples"]
    labels: List[str] = balanced["labels"]

    eval_items: List[Dict] = []
    n = min(len(samples), len(labels))

    for i in range(n):
        text = samples[i]
        label = labels[i]
        polarity = infer_polarity(label)

        item = {
            "text": text,
            "main_emotion": label,
            "polarity": polarity,
            "source": "balanced_training_samples",
            "id": f"balanced_{i:05d}"
        }
        eval_items.append(item)

    # 写入 eval_set.json
    os.makedirs(os.path.dirname(EVAL_PATH), exist_ok=True)
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已基于 balanced_training_samples.json 生成评估集: {EVAL_PATH}")
    print(f"[INFO] 样本数: {len(eval_items)}")


if __name__ == "__main__":
    build_eval_set()
