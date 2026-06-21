#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 eval_set.json 上评估不同版本的级联情绪模型

用法示例：
    python 08_工具脚本\evaluate_cascade_models.py

说明：
- 评估数据来自 05_数据文件\eval_set.json
- 默认会尝试加载这些模型（如果存在）：
    1) d:\silent like onion\emotion_predictor\models\cascade_model_base.pkl
    2) d:\silent like onion\emotion_predictor\models\cascade_model_final.pkl
- 你也可以在脚本里自行修改 MODEL_PATHS 列表，加入你从 Colab 下载的其它版本。

输出：
- 对每个模型，打印：
    - 极性准确率 (polarity_accuracy)
    - 主情绪准确率 (main_emotion_accuracy)
    - 样本总数
"""

import json
import os
import sys
from typing import List, Dict, Any

# 路径配置
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
EVAL_PATH = str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json")
PROJECT_ROOT = r"d:\\silent like onion"
EMOTION_PREDICTOR_DIR = os.path.join(PROJECT_ROOT, "emotion_predictor")

# 默认要评估的模型列表（可自行扩展/修改）
MODEL_PATHS = [
    os.path.join(EMOTION_PREDICTOR_DIR, "models", "cascade_model_base.pkl"),
    os.path.join(EMOTION_PREDICTOR_DIR, "models", "cascade_model_final.pkl"),
]


def load_eval_set(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到评估集: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_set.json 的顶层结构应为列表[list]")
    return data


def ensure_import_path() -> None:
    """确保可以导入 emotion_predictor.cascade_predictor"""
    if EMOTION_PREDICTOR_DIR not in sys.path:
        sys.path.append(EMOTION_PREDICTOR_DIR)


def evaluate_model(model_path: str, eval_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    ensure_import_path()
    try:
        from cascade_predictor import CascadePredictor
    except ImportError as e:
        raise ImportError(f"无法导入 cascade_predictor: {e}")

    if not os.path.exists(model_path):
        return {
            "model_path": model_path,
            "exists": False,
            "polarity_accuracy": None,
            "main_emotion_accuracy": None,
            "samples": 0,
        }

    predictor = CascadePredictor(model_path=model_path, device="cpu")

    total = 0
    correct_polarity = 0
    correct_emotion = 0

    for item in eval_set:
        text = item.get("text", "")
        true_pol = item.get("polarity")
        true_emo = item.get("main_emotion")
        if not text or true_pol is None or true_emo is None:
            continue

        try:
            pred = predictor.predict_single(text)
        except Exception:
            # 单条失败就跳过，避免影响整体评估
            continue

        pred_pol = pred.get("polarity")
        pred_emo = pred.get("main_emotion")

        total += 1
        if pred_pol == true_pol:
            correct_polarity += 1
        if pred_emo == true_emo:
            correct_emotion += 1

    if total == 0:
        polarity_acc = None
        emotion_acc = None
    else:
        polarity_acc = correct_polarity / total
        emotion_acc = correct_emotion / total

    return {
        "model_path": model_path,
        "exists": True,
        "polarity_accuracy": polarity_acc,
        "main_emotion_accuracy": emotion_acc,
        "samples": total,
    }


def main() -> None:
    print("=" * 80)
    print("EVALUATE CASCADE MODELS ON EVAL SET")
    print("=" * 80)

    eval_set = load_eval_set(EVAL_PATH)
    print(f"[INFO] 评估样本数: {len(eval_set)}")
    print()

    results: List[Dict[str, Any]] = []
    for path in MODEL_PATHS:
        print(f"[INFO] 评估模型: {path}")
        res = evaluate_model(path, eval_set)
        results.append(res)

    print("\n结果汇总:")
    print("-" * 80)
    header = f"{'Model':50s}  {'Exists':6s}  {'PolAcc':8s}  {'EmoAcc':8s}  {'N':5s}"
    print(header)
    print("-" * 80)

    for r in results:
        model_name = os.path.basename(r["model_path"])
        exists = "Y" if r["exists"] else "N"
        if r["polarity_accuracy"] is None:
            pol_str = "-"
            emo_str = "-"
        else:
            pol_str = f"{r['polarity_accuracy']:.3f}"
            emo_str = f"{r['main_emotion_accuracy']:.3f}"
        n_str = str(r["samples"])
        line = f"{model_name:50s}  {exists:6s}  {pol_str:8s}  {emo_str:8s}  {n_str:5s}"
        print(line)

    print("-" * 80)
    print("[INFO] 评估完成。你可以根据上述结果选择哪一个模型作为当前主版本。")


if __name__ == "__main__":
    main()
