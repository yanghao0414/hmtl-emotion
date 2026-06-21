#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细评估脚本 - 生成论文所需的所有指标
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

bootstrap()

import json
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from hmtl_evaluate import HMTLPredictor
from hmtl_utils import LABEL_4_NAMES, LABEL_3_NAMES

def detailed_evaluation(model_path, eval_data_path):
    """
    详细评估模型性能
    """
    print("="*70)
    print("HMTL V2 模型详细评估")
    print("="*70)
    
    # 加载模型
    print("\n[1] 加载模型...")
    predictor = HMTLPredictor(model_path)
    
    # 加载评估数据
    print("\n[2] 加载评估数据...")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    total = len(eval_data)
    print(f"✓ 评估样本数: {total}")
    
    # 统计变量
    correct_4 = 0
    correct_3 = 0
    correct_7 = 0
    
    # 收集预测和真实值
    true_arousal = []
    pred_arousal = []
    true_valence = []
    pred_valence = []
    
    # 混淆记录
    confusion_7 = []
    
    # 详细记录
    predictions = []
    
    print("\n[3] 开始评估...")
    for i, item in enumerate(eval_data, 1):
        if i % 50 == 0:
            print(f"  进度: {i}/{total}")
        
        text = item['text']
        true_emotion = item['original_emotion']
        true_label_4 = item['label_4']
        true_label_3 = item['label_3']
        true_arou = item['arousal']
        true_vale = item['valence']
        
        # 预测
        pred = predictor.predict(text, return_details=False)
        
        # 4分类
        pred_label_4 = list(LABEL_4_NAMES.keys())[
            list(LABEL_4_NAMES.values()).index(pred['emotion_4'])
        ]
        if pred_label_4 == true_label_4:
            correct_4 += 1
        
        # 3分类
        pred_label_3 = list(LABEL_3_NAMES.keys())[
            list(LABEL_3_NAMES.values()).index(pred['polarity_3'])
        ]
        if pred_label_3 == true_label_3:
            correct_3 += 1
        
        # 7类
        if pred['emotion_7'] == true_emotion:
            correct_7 += 1
        else:
            confusion_7.append((true_emotion, pred['emotion_7']))
        
        # 回归
        true_arousal.append(true_arou)
        pred_arousal.append(pred['arousal'])
        true_valence.append(true_vale)
        pred_valence.append(pred['valence'])
        
        # 记录
        predictions.append({
            'text': text,
            'true': true_emotion,
            'pred': pred['emotion_7'],
            'correct': pred['emotion_7'] == true_emotion
        })
    
    # ========== 计算指标 ==========
    
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    # 1. 分类准确率
    acc_7 = correct_7 / total
    acc_4 = correct_4 / total
    acc_3 = correct_3 / total
    
    error_7 = total - correct_7
    error_4 = total - correct_4
    error_3 = total - correct_3
    
    print("\n【1. 核心分类性能】")
    print(f"\n7类情绪分类:")
    print(f"  准确率: {acc_7:.2%} ({correct_7}/{total})")
    print(f"  错误数: {error_7}")
    
    print(f"\n4核心分类:")
    print(f"  准确率: {acc_4:.2%} ({correct_4}/{total})")
    print(f"  错误数: {error_4}")
    
    print(f"\n3极性分类:")
    print(f"  准确率: {acc_3:.2%} ({correct_3}/{total})")
    print(f"  错误数: {error_3}")
    
    # 2. 回归性能
    true_arousal = np.array(true_arousal)
    pred_arousal = np.array(pred_arousal)
    true_valence = np.array(true_valence)
    pred_valence = np.array(pred_valence)
    
    r2_arousal = r2_score(true_arousal, pred_arousal)
    r2_valence = r2_score(true_valence, pred_valence)
    
    corr_arousal, _ = pearsonr(true_arousal, pred_arousal)
    corr_valence, _ = pearsonr(true_valence, pred_valence)
    
    print("\n【2. 维度回归性能】")
    print(f"\nArousal (唤醒度):")
    print(f"  R² Score: {r2_arousal:.4f}")
    print(f"  Pearson Corr: {corr_arousal:.4f}")
    
    print(f"\nValence (效价):")
    print(f"  R² Score: {r2_valence:.4f}")
    print(f"  Pearson Corr: {corr_valence:.4f}")
    
    # 3. Top 3 混淆
    confusion_counter = Counter(confusion_7)
    top3_confusions = confusion_counter.most_common(3)
    
    print("\n【3. Top 3 混淆分析】")
    for i, ((true_emo, pred_emo), count) in enumerate(top3_confusions, 1):
        print(f"\n混淆 {i}: {true_emo} → {pred_emo} ({count} 次)")
    
    # 如果不足3个，显示所有
    if len(top3_confusions) < 3:
        print(f"\n注: 只有 {len(top3_confusions)} 种混淆情况")
    
    # 4. 训练参数（从训练历史读取）
    print("\n【4. 训练关键参数】")
    try:
        history_path = r"d:\bigcreate\06_模型文件\hmtl_models_v2\training_history_v2.json"
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        total_epochs = len(history)
        print(f"\n总训练轮数: {total_epochs} epochs")
        print(f"Arousal 损失权重 (λ_A): 0.8")
        print(f"Valence 损失权重 (λ_V): 0.5")
        print(f"学习率: 2e-5")
        print(f"Batch Size: 16")
    except:
        print("\n训练历史文件未找到")
    
    print("\n" + "="*70)
    
    # 返回结果
    return {
        'acc_7': acc_7,
        'acc_4': acc_4,
        'acc_3': acc_3,
        'error_7': error_7,
        'error_4': error_4,
        'error_3': error_3,
        'r2_arousal': r2_arousal,
        'r2_valence': r2_valence,
        'corr_arousal': corr_arousal,
        'corr_valence': corr_valence,
        'top3_confusions': top3_confusions
    }


if __name__ == "__main__":
    model_path = r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt"
    eval_data_path = r"d:\bigcreate\05_数据文件\eval_set_hmtl.json"
    
    results = detailed_evaluation(model_path, eval_data_path)
    
    print("\n" + "="*70)
    print("评估完成！")
    print("="*70)
