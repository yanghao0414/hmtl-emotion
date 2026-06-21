#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试模型 - 只测试100条样本"""

import json
import sys
import os

PROJECT_ROOT = r"d:\silent like onion"
EMOTION_PREDICTOR_DIR = os.path.join(PROJECT_ROOT, "emotion_predictor")
sys.path.append(EMOTION_PREDICTOR_DIR)

from cascade_predictor import CascadePredictor

# 加载评估集（只取前100条）
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]

with open(str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json"), "r", encoding="utf-8") as f:
    eval_set = json.load(f)[:100]  # 只测试100条

print(f"测试样本数: {len(eval_set)}")
print("正在加载模型...")

# 初始化预测器
model_path = os.path.join(PROJECT_ROOT, "emotion_predictor", "models", "cascade_model_final.pkl")
predictor = CascadePredictor(model_path=model_path, device="cpu")

print("开始预测...")
correct_pol = 0
correct_emo = 0
total = 0

for i, item in enumerate(eval_set, 1):
    text = item.get("text", "")
    true_pol = item.get("polarity")
    true_emo = item.get("main_emotion")
    
    if not text or true_pol is None or true_emo is None:
        continue
    
    try:
        pred = predictor.predict_single(text)
        pred_pol = pred.get("polarity")
        pred_emo = pred.get("main_emotion")
        
        total += 1
        if pred_pol == true_pol:
            correct_pol += 1
        if pred_emo == true_emo:
            correct_emo += 1
            
        if i % 20 == 0:
            print(f"  已处理: {i}/100")
            
    except Exception as e:
        print(f"  样本 {i} 预测失败: {e}")
        continue

print("\n" + "="*50)
print("快速测试结果 (100条样本)")
print("="*50)
print(f"极性准确率: {correct_pol}/{total} = {correct_pol/total*100:.2f}%")
print(f"情绪准确率: {correct_emo}/{total} = {correct_emo/total*100:.2f}%")
print("="*50)
