#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 eval_set.json 中的细情绪标签映射到 7 大主情绪

效果：
- 保留你原来的细情绪信息到 fine_emotion 字段；
- 用映射表推断 7 大主情绪，写入/覆盖 main_emotion 字段；

用法：
    python 08_工具脚本\remap_eval_emotions.py

前置条件：
- 05_数据文件\eval_set.json 已由 build_eval_from_balanced.py 生成，
  当前的 main_emotion 里存的仍是细标签（如 "理解"、"沮丧" 等）。

7 大主情绪统一为：
- 焦虑 / 悲伤 / 失望 / 愤怒 / 支持 / 快乐 / 平静
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json")

# 细标签 -> 7 大主情绪 的映射表
# 可以根据需要逐步扩充/微调
FINE_TO_MAIN: Dict[str, str] = {
    # 支持 / 安慰 / 理解 / 鼓励 / 放松
    "理解": "支持",
    "安慰": "支持",
    "鼓励": "支持",
    "支持": "支持",
    "放松": "支持",

    # 悲伤 / 沮丧 / 无助
    "悲伤": "悲伤",
    "沮丧": "悲伤",
    "无助": "悲伤",

    # 焦虑 / 紧张 / 担心 / 害怕 / 恐惧 / 困惑 / 犹豫
    "焦虑": "焦虑",
    "紧张": "焦虑",
    "担心": "焦虑",
    "害怕": "焦虑",
    "恐惧": "焦虑",
    "困惑": "焦虑",
    "犹豫": "焦虑",

    # 失望
    "失望": "失望",

    # 愤怒相关
    "愤怒": "愤怒",
    "生气": "愤怒",

    # 快乐 / 兴奋 / 期待 / 有希望 / 自信
    "快乐": "快乐",
    "高兴": "快乐",
    "开心": "快乐",
    "兴奋": "快乐",
    "激动": "快乐",
    "期待": "快乐",
    "希望": "快乐",
    "有希望": "快乐",
    "自信": "快乐",

    # 平静
    "平静": "平静",
}

# 对于未出现在映射表中的细标签，默认归为哪一类
DEFAULT_MAIN = "平静"


def map_fine_to_main(fine: str) -> str:
    """将细情绪映射到 7 大主情绪。"""
    if not isinstance(fine, str):
        return DEFAULT_MAIN
    fine = fine.strip()
    if fine in FINE_TO_MAIN:
        return FINE_TO_MAIN[fine]
    # 可以在这里加一些简单规则，比如包含某些关键词时归类
    # 目前先严格使用字典匹配，未命中的一律归到 DEFAULT_MAIN
    return DEFAULT_MAIN


def remap_eval_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到评估集文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval_set.json 顶层结构必须是列表[list]")

    updated: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        # 原来的 main_emotion 先当成细情绪备份
        old_main = item.get("main_emotion")
        if "fine_emotion" not in item:
            item["fine_emotion"] = old_main

        fine = item.get("fine_emotion", old_main)
        main_7 = map_fine_to_main(fine)
        item["main_emotion"] = main_7

        updated.append(item)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已更新评估集: {path}")
    print(f"[INFO] 样本数: {len(updated)}")


if __name__ == "__main__":
    remap_eval_file(EVAL_PATH)
