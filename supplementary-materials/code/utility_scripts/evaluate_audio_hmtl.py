#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估 Audio HMTL V2 模型的 7类/4类/3类 准确率
使用缓存的特征进行快速评估
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import Wav2Vec2Model, Wav2Vec2Config

# 标签名称
LABEL_7_NAMES = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
LABEL_4_NAMES = ['积极', '激活消极', '非激活消极', '平静']
LABEL_3_NAMES = ['积极', '消极', '中性']

# 路径
MODEL_PATH = r"d:\bigcreate\06_模型文件\audio_hmtl_v2_best.pt"
LABELS_PATH = r"d:\bigcreate\05_数据文件\audio_hmtl_labels_v2.csv"
CACHE_PATH = r"d:\bigcreate\05_数据文件\audio_features_cache.pt"


class AudioHMTLClassifier(nn.Module):
    """Audio HMTL V2 模型"""
    
    def __init__(self, dropout=0.3):
        super().__init__()
        
        WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
        self.wav2vec2.freeze_feature_extractor()
        
        config = Wav2Vec2Config.from_pretrained(WAV2VEC2_MODEL_NAME)
        hidden_size = config.hidden_size  # 768
        
        self.dim_reducer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier_7 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 7)
        )
        
        self.classifier_4 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(64, 4)
        )
        
        self.classifier_3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(64, 3)
        )
        
        self.regressor_A = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.regressor_V = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, input_values, attention_mask=None):
        output = self.wav2vec2(input_values, attention_mask=attention_mask)
        mean_pooling = torch.mean(output.last_hidden_state, dim=1)
        x = self.dim_reducer(mean_pooling)
        
        return {
            'label_7_logits': self.classifier_7(x),
            'label_4_logits': self.classifier_4(x),
            'label_3_logits': self.classifier_3(x),
            'arousal': self.regressor_A(x).squeeze(-1),
            'valence': self.regressor_V(x).squeeze(-1)
        }
    
    def forward_from_features(self, features):
        """从已提取的特征进行前向传播"""
        x = self.dim_reducer(features)
        return {
            'label_7_logits': self.classifier_7(x),
            'label_4_logits': self.classifier_4(x),
            'label_3_logits': self.classifier_3(x),
            'arousal': self.regressor_A(x).squeeze(-1),
            'valence': self.regressor_V(x).squeeze(-1)
        }


def load_model(model_path, device):
    """加载模型"""
    print(f"加载模型: {model_path}")
    model = AudioHMTLClassifier()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_with_cache():
    """使用缓存特征评估"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    model = load_model(MODEL_PATH, device)
    
    # 加载标签
    df = pd.read_csv(LABELS_PATH)
    print(f"标签文件样本数: {len(df)}")
    
    # 检查是否有缓存
    if os.path.exists(CACHE_PATH):
        print(f"加载特征缓存: {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, map_location=device)
        
        # 检查缓存结构
        print(f"缓存 keys: {list(cache.keys())[:5]}...")
        
        # 匹配标签和特征
        y_true_7, y_true_4, y_true_3 = [], [], []
        y_pred_7, y_pred_4, y_pred_3 = [], [], []
        
        matched = 0
        for idx, row in df.iterrows():
            audio_path = row['audio_full_path']
            
            # 尝试不同的 key 格式
            feature = None
            for key in [audio_path, os.path.basename(audio_path), audio_path.replace('\\', '/')]:
                if key in cache:
                    feature = cache[key]
                    break
            
            if feature is None:
                continue
            
            matched += 1
            
            # 获取标签
            label_7 = row['label_7_emotion']
            label_4 = row['label_4_emotion']
            label_3 = row['label_3_polarity']
            
            # 预测
            with torch.no_grad():
                if isinstance(feature, torch.Tensor):
                    feat = feature.unsqueeze(0).to(device) if feature.dim() == 1 else feature.to(device)
                else:
                    feat = torch.tensor(feature).unsqueeze(0).to(device)
                
                outputs = model.forward_from_features(feat)
                
                pred_7 = outputs['label_7_logits'].argmax(dim=1).item()
                pred_4 = outputs['label_4_logits'].argmax(dim=1).item()
                pred_3 = outputs['label_3_logits'].argmax(dim=1).item()
            
            y_true_7.append(label_7)
            y_pred_7.append(pred_7)
            y_true_4.append(label_4)
            y_pred_4.append(pred_4)
            y_true_3.append(label_3)
            y_pred_3.append(pred_3)
        
        print(f"匹配样本数: {matched}")
        
        if matched == 0:
            print("无法匹配特征，尝试直接使用缓存...")
            # 直接遍历缓存
            return evaluate_cache_directly(model, cache, df, device)
        
        # 计算准确率
        acc_7 = accuracy_score(y_true_7, y_pred_7) * 100
        acc_4 = accuracy_score(y_true_4, y_pred_4) * 100
        acc_3 = accuracy_score(y_true_3, y_pred_3) * 100
        
        print("\n" + "="*50)
        print("Audio HMTL V2 评估结果")
        print("="*50)
        print(f"7类准确率: {acc_7:.2f}%")
        print(f"4类准确率: {acc_4:.2f}%")
        print(f"3类准确率: {acc_3:.2f}%")
        
        return acc_7, acc_4, acc_3, y_true_7, y_pred_7, y_true_4, y_pred_4, y_true_3, y_pred_3
    else:
        print("特征缓存不存在，无法评估")
        return None


def evaluate_cache_directly(model, cache, df, device):
    """直接使用缓存评估"""
    # 检查缓存格式
    sample_key = list(cache.keys())[0]
    sample_val = cache[sample_key]
    print(f"缓存样本 key: {sample_key}")
    print(f"缓存样本 value type: {type(sample_val)}")
    
    if isinstance(sample_val, dict):
        print(f"缓存样本 value keys: {sample_val.keys()}")
    
    return None


if __name__ == '__main__':
    result = evaluate_with_cache()
    
    if result:
        acc_7, acc_4, acc_3, y_true_7, y_pred_7, y_true_4, y_pred_4, y_true_3, y_pred_3 = result
        
        print("\n" + "="*50)
        print("7类分类报告")
        print("="*50)
        print(classification_report(y_true_7, y_pred_7, target_names=LABEL_7_NAMES, zero_division=0))
        
        print("\n" + "="*50)
        print("4类分类报告")
        print("="*50)
        print(classification_report(y_true_4, y_pred_4, target_names=LABEL_4_NAMES, zero_division=0))
