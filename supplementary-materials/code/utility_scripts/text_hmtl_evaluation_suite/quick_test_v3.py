#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text HMTL V3 模型快速验证
使用56条手动标注数据测试真实性能
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix

# 4类和7类名称
LABEL_4_NAMES = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}

# 7类→4类映射
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}
LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}


class TextHMTLModelV3(nn.Module):
    """Text HMTL V3 模型定义"""
    
    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.3):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.classifier_4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, 4)
        )
        
        self.classifier_3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 3)
        )
        
        self.classifier_7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, 7)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        return {
            'label_4_logits': self.classifier_4(cls_output),
            'label_3_logits': self.classifier_3(cls_output),
            'label_7_logits': self.classifier_7(cls_output),
            'arousal': self.arousal_head(cls_output).squeeze(-1),
            'valence': self.valence_head(cls_output).squeeze(-1)
        }


def load_model(model_path, device):
    """加载V3模型"""
    print(f"加载模型: {model_path}")
    
    model = TextHMTLModelV3()
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"训练时4类准确率: {checkpoint.get('acc_4', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def predict_batch(model, tokenizer, texts, device):
    """批量预测"""
    preds_4, preds_3, preds_7 = [], [], []
    
    for text in texts:
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        
        preds_4.append(outputs['label_4_logits'].argmax(dim=1).item())
        preds_3.append(outputs['label_3_logits'].argmax(dim=1).item())
        preds_7.append(outputs['label_7_logits'].argmax(dim=1).item())
    
    return preds_7, preds_4, preds_3


def main():
    print("="*60)
    print("Text HMTL V3 模型验证")
    print("="*60)
    
    # 1. 加载数据
    data_path = "data/raw/evaluation_dataset.csv"
    print(f"\n加载数据: {data_path}")
    df = pd.read_csv(data_path)
    df_manual = df[df['source'] == 'manual'].copy()
    print(f"手动标注数据: {len(df_manual)} 条")
    
    # 2. 加载模型
    model_path = r"d:\bigcreate\06_模型文件\text_hmtl_v3_best.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 3. 预测
    print("\n开始预测...")
    texts = df_manual['text'].tolist()
    y_true_7 = df_manual['label_7'].astype(int).tolist()
    
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
    print("\n【4类分类详情】")
    for label_id in range(4):
        mask = [t == label_id for t in y_true_4]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_4, y_pred_4) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {LABEL_4_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 7类详情
    print("\n【7类分类详情】")
    for label_id in range(7):
        mask = [t == label_id for t in y_true_7]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_7, y_pred_7) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {EMOTION_7_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 4类混淆矩阵
    print("\n【4类混淆矩阵】")
    cm_4 = confusion_matrix(y_true_4, y_pred_4, labels=range(4))
    print("        " + "  ".join([f"{LABEL_4_NAMES[i][:2]:>4}" for i in range(4)]))
    for i, row in enumerate(cm_4):
        row_str = "  ".join([f"{v:4d}" for v in row])
        print(f"{LABEL_4_NAMES[i][:4]:>6}  {row_str}")
    
    # 错误样本
    print("\n" + "="*60)
    print("4类错误样本 (最多10个)")
    print("="*60)
    
    errors = []
    for i, (text, true, pred) in enumerate(zip(texts, y_true_4, y_pred_4)):
        if true != pred:
            errors.append({
                'text': text[:40],
                'true': LABEL_4_NAMES[true],
                'pred': LABEL_4_NAMES[pred]
            })
    
    for e in errors[:10]:
        print(f"  文本: {e['text']}...")
        print(f"  真实: {e['true']} → 预测: {e['pred']}")
        print()
    
    # 结论
    print("="*60)
    print("V2 vs V3 对比")
    print("="*60)
    print(f"         V2        V3")
    print(f"4类:    60.7%  →  {acc_4*100:.1f}%")
    print(f"7类:    60.7%  →  {acc_7*100:.1f}%")


if __name__ == '__main__':
    main()
