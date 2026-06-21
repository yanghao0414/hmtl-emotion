#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证 Text HMTL 模型
用已下载的数据直接测试，看模型在新数据上的表现
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '02_模型代码'))

from transformers import BertTokenizer

EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}


def load_model(model_path):
    """加载模型"""
    from hmtl_model_v2 import HMTLEmotionModelV2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    model = HMTLEmotionModelV2()
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device


def predict_batch(model, tokenizer, texts, device, batch_size=16):
    """批量预测"""
    pred_7_list = []
    pred_4_list = []
    pred_3_list = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encoding = tokenizer(
            batch_texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        pred_7 = outputs['label_7_logits'].argmax(dim=1).cpu().numpy()
        pred_4 = outputs['label_4_logits'].argmax(dim=1).cpu().numpy()
        pred_3 = outputs['label_3_logits'].argmax(dim=1).cpu().numpy()
        
        pred_7_list.extend(pred_7)
        pred_4_list.extend(pred_4)
        pred_3_list.extend(pred_3)
    
    return pred_7_list, pred_4_list, pred_3_list


def main():
    print("="*60)
    print("Text HMTL 模型快速验证")
    print("="*60)
    
    # 1. 加载数据
    data_path = "data/raw/evaluation_dataset.csv"
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 只用手动标注的56条（这些标签是准确的）
    df_manual = df[df['source'] == 'manual'].copy()
    print(f"使用手动标注数据: {len(df_manual)} 条")
    
    # 2. 加载模型
    model_path = r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt"
    print(f"\n加载模型: {model_path}")
    
    model, device = load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("模型加载完成!")
    
    # 3. 预测
    print("\n开始预测...")
    texts = df_manual['text'].tolist()
    y_true_7 = df_manual['label_7'].astype(int).tolist()
    
    # 7类→4类映射: 愤怒/焦虑→激活消极(1), 快乐/支持→积极(0), 悲伤/失望→非激活消极(2), 平静→平静(3)
    LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}
    # 7类→3类映射: 愤怒/焦虑/悲伤/失望→消极(1), 快乐/支持→积极(0), 平静→平静(2)
    LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}
    
    y_true_4 = [LABEL_7_TO_4[y] for y in y_true_7]
    y_true_3 = [LABEL_7_TO_3[y] for y in y_true_7]
    
    y_pred_7, y_pred_4, y_pred_3 = predict_batch(model, tokenizer, texts, device)
    
    # 4. 评估
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    acc_7 = accuracy_score(y_true_7, y_pred_7)
    acc_4 = accuracy_score(y_true_4, y_pred_4)
    acc_3 = accuracy_score(y_true_3, y_pred_3)
    
    print(f"\n┌─────────────────────────────────┐")
    print(f"│  3类准确率: {acc_3*100:6.2f}%            │")
    print(f"│  4类准确率: {acc_4*100:6.2f}%            │")
    print(f"│  7类准确率: {acc_7*100:6.2f}%            │")
    print(f"└─────────────────────────────────┘")
    
    # 4类详情
    LABEL_4_NAMES = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}
    print("\n【4类分类详情】")
    for label_id in range(4):
        mask = [t == label_id for t in y_true_4]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_4, y_pred_4) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {LABEL_4_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 7类每类准确率
    print("\n【7类分类详情】")
    for label_id in range(7):
        mask = [t == label_id for t in y_true_7]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_7, y_pred_7) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {EMOTION_7_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 混淆矩阵
    print("\n【7类混淆矩阵】")
    cm = confusion_matrix(y_true_7, y_pred_7, labels=range(7))
    print("        " + "  ".join([f"{EMOTION_7_NAMES[i][:2]:>4}" for i in range(7)]))
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:4d}" for v in row])
        print(f"{EMOTION_7_NAMES[i][:2]:>6}  {row_str}")
    
    # 错误样本分析
    print("\n" + "="*60)
    print("错误样本分析 (最多显示10个)")
    print("="*60)
    
    errors = []
    for i, (text, true, pred) in enumerate(zip(texts, y_true_7, y_pred_7)):
        if true != pred:
            errors.append({
                'text': text[:50],
                'true': EMOTION_7_NAMES[true],
                'pred': EMOTION_7_NAMES[pred]
            })
    
    for e in errors[:10]:
        print(f"  文本: {e['text']}...")
        print(f"  真实: {e['true']} → 预测: {e['pred']}")
        print()
    
    # 结论
    print("="*60)
    print("结论")
    print("="*60)
    
    if acc >= 0.85:
        print(f"✅ 准确率 {acc*100:.1f}% - 模型表现优秀!")
    elif acc >= 0.70:
        print(f"✅ 准确率 {acc*100:.1f}% - 模型表现良好")
    elif acc >= 0.55:
        print(f"⚠️ 准确率 {acc*100:.1f}% - 模型表现一般，需要优化")
    else:
        print(f"❌ 准确率 {acc*100:.1f}% - 模型可能过拟合，需要重新训练")
    
    print("\n注意: 这是在56条手动样本上的测试结果")
    print("如需更准确的评估，请标注更多数据后重新测试")


if __name__ == '__main__':
    main()
