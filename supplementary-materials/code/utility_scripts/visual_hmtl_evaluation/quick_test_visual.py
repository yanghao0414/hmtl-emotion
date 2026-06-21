#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual HMTL 模型快速验证
"""

import os
import sys
import random
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

# 标签名称
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}
LABEL_4_NAMES = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}


class VisualHMTLClassifier(nn.Module):
    """Visual HMTL 分类器"""
    
    def __init__(self, dropout=0.3):
        super().__init__()
        
        self.backbone = models.efficientnet_b2(weights=None)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 7)
        )
        
        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4)
        )
        
        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 3)
        )
        
        self.regressor_A = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.regressor_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        return {
            'label_7_logits': self.classifier_7(shared),
            'label_4_logits': self.classifier_4(shared),
            'label_3_logits': self.classifier_3(shared),
            'arousal': self.regressor_A(shared).squeeze(-1),
            'valence': self.regressor_V(shared).squeeze(-1)
        }


def load_model(model_path, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    model = VisualHMTLClassifier()
    
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    print("="*60)
    print("Visual HMTL 模型验证")
    print("="*60)
    
    # 1. 加载标签数据
    labels_path = r"d:\bigcreate\05_数据文件\labels\visual_hmtl_labels.csv"
    print(f"\n加载标签: {labels_path}")
    df = pd.read_csv(labels_path)
    print(f"总样本数: {len(df)}")
    
    # 2. 随机抽取测试样本（从 Test 目录）
    df_test = df[df['image_path'].str.contains('Test')]
    print(f"测试集样本: {len(df_test)}")
    
    # 每类抽取一些样本
    sample_per_class = 50
    test_samples = []
    for label_7 in range(7):
        class_df = df_test[df_test['label_7'] == label_7]
        if len(class_df) > 0:
            n = min(sample_per_class, len(class_df))
            samples = class_df.sample(n=n, random_state=42)
            test_samples.append(samples)
            print(f"  {EMOTION_7_NAMES[label_7]}: 抽取 {n} 条")
    
    df_sample = pd.concat(test_samples, ignore_index=True)
    print(f"\n总测试样本: {len(df_sample)}")
    
    # 3. 加载模型
    model_path = r"d:\bigcreate\06_模型文件\visual_hmtl_best.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    model = load_model(model_path, device)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. 预测
    print("\n开始预测...")
    y_true_7, y_pred_7 = [], []
    y_true_4, y_pred_4 = [], []
    y_true_3, y_pred_3 = [], []
    
    valid_count = 0
    for idx, row in df_sample.iterrows():
        img_path = row['image_path']
        
        if not os.path.exists(img_path):
            continue
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
            
            pred_7 = outputs['label_7_logits'].argmax(dim=1).item()
            pred_4 = outputs['label_4_logits'].argmax(dim=1).item()
            pred_3 = outputs['label_3_logits'].argmax(dim=1).item()
            
            y_true_7.append(row['label_7'])
            y_pred_7.append(pred_7)
            y_true_4.append(row['label_4'])
            y_pred_4.append(pred_4)
            y_true_3.append(row['label_3'])
            y_pred_3.append(pred_3)
            
            valid_count += 1
        except Exception as e:
            continue
    
    print(f"有效样本: {valid_count}")
    
    # 5. 评估
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    acc_7 = accuracy_score(y_true_7, y_pred_7)
    acc_4 = accuracy_score(y_true_4, y_pred_4)
    acc_3 = accuracy_score(y_true_3, y_pred_3)
    
    print(f"\n┌─────────────────────────────────┐")
    print(f"│  7类准确率: {acc_7*100:6.2f}%            │")
    print(f"│  4类准确率: {acc_4*100:6.2f}%            │")
    print(f"│  3类准确率: {acc_3*100:6.2f}%            │")
    print(f"└─────────────────────────────────┘")
    
    # 7类详情
    print("\n【7类分类详情】")
    for label_id in range(7):
        mask = [t == label_id for t in y_true_7]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_7, y_pred_7) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {EMOTION_7_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 4类详情
    print("\n【4类分类详情】")
    for label_id in range(4):
        mask = [t == label_id for t in y_true_4]
        if sum(mask) > 0:
            correct = sum(1 for t, p in zip(y_true_4, y_pred_4) if t == label_id and t == p)
            total = sum(mask)
            class_acc = correct / total
            print(f"  {LABEL_4_NAMES[label_id]}: {correct}/{total} = {class_acc*100:.1f}%")
    
    # 7类混淆矩阵
    print("\n【7类混淆矩阵】")
    cm_7 = confusion_matrix(y_true_7, y_pred_7, labels=range(7))
    print("        " + "  ".join([f"{EMOTION_7_NAMES[i][:2]:>4}" for i in range(7)]))
    for i, row in enumerate(cm_7):
        row_str = "  ".join([f"{v:4d}" for v in row])
        print(f"{EMOTION_7_NAMES[i][:2]:>6}  {row_str}")


if __name__ == '__main__':
    main()
