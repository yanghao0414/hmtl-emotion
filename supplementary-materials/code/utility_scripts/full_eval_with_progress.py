#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整评估模型 - 2554条样本 - 带详细进度"""

import json
import sys
import os
from datetime import datetime

PROJECT_ROOT = r"d:\silent like onion"
EMOTION_PREDICTOR_DIR = os.path.join(PROJECT_ROOT, "emotion_predictor")
sys.path.append(EMOTION_PREDICTOR_DIR)

from cascade_predictor import CascadePredictor

print("="*60)
print("完整模型评估 - cascade_model_final.pkl")
print("="*60)

# 加载评估集
print("\n[1/4] 加载评估集...")
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]

with open(str(_PROJECT_ROOT / "05_数据文件" / "eval_set.json"), "r", encoding="utf-8") as f:
    eval_set = json.load(f)
print(f"✓ 已加载 {len(eval_set)} 条样本")

# 初始化预测器
print("\n[2/4] 加载模型和BERT...")
model_path = os.path.join(PROJECT_ROOT, "emotion_predictor", "models", "cascade_model_final.pkl")
predictor = CascadePredictor(model_path=model_path, device="cpu")
print("✓ 模型加载完成")

# 开始预测
print(f"\n[3/4] 开始预测 {len(eval_set)} 条样本...")
print("(大约需要 3-5 分钟)\n")

start_time = datetime.now()
correct_pol = 0
correct_emo = 0
total = 0
errors = []

# 情绪混淆统计
confusion = {}

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
        
        # 极性准确率
        if pred_pol == true_pol:
            correct_pol += 1
        
        # 情绪准确率
        if pred_emo == true_emo:
            correct_emo += 1
        else:
            # 记录混淆
            key = f"{true_emo} → {pred_emo}"
            confusion[key] = confusion.get(key, 0) + 1
        
        # 每500条显示进度
        if i % 500 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            speed = i / elapsed if elapsed > 0 else 0
            eta = (len(eval_set) - i) / speed if speed > 0 else 0
            print(f"  进度: {i}/{len(eval_set)} ({i/len(eval_set)*100:.1f}%) | "
                  f"速度: {speed:.1f}条/秒 | ETA: {eta/60:.1f}分钟")
            
    except Exception as e:
        errors.append({"index": i, "text": text[:50], "error": str(e)})
        continue

elapsed_time = (datetime.now() - start_time).total_seconds()

# 结果输出
print("\n" + "="*60)
print("[4/4] 评估完成！")
print("="*60)
print(f"\n总样本数: {total}")
print(f"处理时间: {elapsed_time/60:.1f} 分钟")
print(f"处理速度: {total/elapsed_time:.1f} 条/秒")

print("\n【准确率】")
print(f"  极性准确率: {correct_pol}/{total} = {correct_pol/total*100:.2f}%")
print(f"  情绪准确率: {correct_emo}/{total} = {correct_emo/total*100:.2f}%")

if errors:
    print(f"\n【错误】")
    print(f"  失败样本数: {len(errors)}")

# Top 10 混淆
if confusion:
    print("\n【Top 10 情绪混淆】")
    sorted_conf = sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]
    for pair, count in sorted_conf:
        print(f"  {pair}: {count}次")

print("\n" + "="*60)

# 保存详细结果
result = {
    "timestamp": datetime.now().isoformat(),
    "model": "cascade_model_final.pkl",
    "total_samples": total,
    "polarity_accuracy": correct_pol/total if total > 0 else 0,
    "emotion_accuracy": correct_emo/total if total > 0 else 0,
    "elapsed_seconds": elapsed_time,
    "errors_count": len(errors),
    "top_confusions": sorted_conf[:10] if confusion else []
}

with open(str(_PROJECT_ROOT / "05_数据文件" / "eval_result_full.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    
print("✓ 详细结果已保存到: eval_result_full.json")
