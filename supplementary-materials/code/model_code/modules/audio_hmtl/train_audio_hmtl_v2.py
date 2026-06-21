#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio HMTL V2 训练脚本
支持 7类情绪输出，与 Text HMTL 统一格式
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 02_模型代码/

from modules.audio_hmtl.audio_hmtl_classifier import AudioHMTLClassifier

# ============================================================
# 配置
# ============================================================
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
LABELS_CSV = str(_PROJECT_ROOT / "05_数据文件" / "audio_hmtl_labels_v2.csv")
CACHE_FILE = str(_PROJECT_ROOT / "05_数据文件" / "audio_features_cache.pt")
MODEL_SAVE_PATH = str(_PROJECT_ROOT / "06_模型文件" / "audio_hmtl_v2_best.pt")

# 7类权重 (根据分布计算)
# [愤怒:168, 焦虑:135, 快乐:768, 悲伤:49, 失望:58, 支持:671, 平静:3062]
CLASS_WEIGHTS_7 = torch.tensor([
    4911 / (7 * 168),   # 愤怒
    4911 / (7 * 135),   # 焦虑
    4911 / (7 * 768),   # 快乐
    4911 / (7 * 49),    # 悲伤
    4911 / (7 * 58),    # 失望
    4911 / (7 * 671),   # 支持
    4911 / (7 * 3062),  # 平静
])

# 4类权重 (沿用之前的)
CLASS_WEIGHTS_4 = torch.tensor([0.8466, 4.0917, 11.2874, 0.4023])


# ============================================================
# 数据集
# ============================================================
class CachedAudioDatasetV2(Dataset):
    """使用缓存的音频数据集，支持7类标签"""
    
    def __init__(self, label_df, cached_features):
        self.cached_features = cached_features
        self.label_df = label_df
        
        # 找出有缓存的样本
        self.valid_indices = []
        for idx in label_df.index:
            if idx in cached_features:
                self.valid_indices.append(idx)
        
        print(f"有效样本数: {len(self.valid_indices)}/{len(label_df)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        row = self.label_df.loc[orig_idx]
        
        features = self.cached_features[orig_idx]
        
        return {
            'features': features,
            'label_7': torch.tensor(row['label_7_emotion'], dtype=torch.long),
            'label_4': torch.tensor(row['label_4_emotion'], dtype=torch.long),
            'label_3': torch.tensor(row['label_3_polarity'], dtype=torch.long),
            'arousal': torch.tensor(row['true_arousal'], dtype=torch.float),
            'valence': torch.tensor(row['true_valence'], dtype=torch.float),
        }


def collate_fn(batch):
    """批次整理函数"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # 随机裁剪到固定长度
    max_len = 48000  # 3秒
    
    features_list = []
    for item in batch:
        feat = item['features']
        if feat.shape[0] > max_len:
            start = np.random.randint(0, feat.shape[0] - max_len)
            feat = feat[start:start + max_len]
        elif feat.shape[0] < max_len:
            pad_len = max_len - feat.shape[0]
            feat = F.pad(feat, (0, pad_len))
        features_list.append(feat)
    
    return {
        'features': torch.stack(features_list),
        'label_7': torch.stack([b['label_7'] for b in batch]),
        'label_4': torch.stack([b['label_4'] for b in batch]),
        'label_3': torch.stack([b['label_3'] for b in batch]),
        'arousal': torch.stack([b['arousal'] for b in batch]),
        'valence': torch.stack([b['valence'] for b in batch]),
    }


# ============================================================
# 损失函数
# ============================================================
class HMTLAudioLossV2(nn.Module):
    """
    Audio HMTL V2 损失函数
    支持 7类分类
    """
    
    def __init__(self, 
                 weight_7=1.0,
                 weight_4=0.5,
                 weight_3=0.3,
                 weight_A=0.1,
                 weight_V=0.1,
                 class_weights_7=None,
                 class_weights_4=None):
        super().__init__()
        
        self.weight_7 = weight_7
        self.weight_4 = weight_4
        self.weight_3 = weight_3
        self.weight_A = weight_A
        self.weight_V = weight_V
        
        self.ce_loss_7 = nn.CrossEntropyLoss(weight=class_weights_7)
        self.ce_loss_4 = nn.CrossEntropyLoss(weight=class_weights_4)
        self.ce_loss_3 = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        loss_7 = self.ce_loss_7(outputs['label_7_logits'], targets['label_7'])
        loss_4 = self.ce_loss_4(outputs['label_4_logits'], targets['label_4'])
        loss_3 = self.ce_loss_3(outputs['label_3_logits'], targets['label_3'])
        loss_A = self.mse_loss(outputs['arousal'], targets['arousal'])
        loss_V = self.mse_loss(outputs['valence'], targets['valence'])
        
        total_loss = (self.weight_7 * loss_7 +
                      self.weight_4 * loss_4 +
                      self.weight_3 * loss_3 +
                      self.weight_A * loss_A +
                      self.weight_V * loss_V)
        
        return {
            'total_loss': total_loss,
            'loss_7': loss_7.item(),
            'loss_4': loss_4.item(),
            'loss_3': loss_3.item(),
            'loss_A': loss_A.item(),
            'loss_V': loss_V.item(),
        }


# ============================================================
# 训练函数
# ============================================================
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct_7 = 0
    correct_4 = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            features = batch['features'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'features'}
            
            outputs = model(features)
            loss_dict = criterion(outputs, targets)
            
            total_loss += loss_dict['total_loss'].item()
            
            pred_7 = outputs['label_7_logits'].argmax(dim=1)
            pred_4 = outputs['label_4_logits'].argmax(dim=1)
            
            correct_7 += (pred_7 == targets['label_7']).sum().item()
            correct_4 += (pred_4 == targets['label_4']).sum().item()
            total += targets['label_7'].size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'acc_7': correct_7 / total if total > 0 else 0,
        'acc_4': correct_4 / total if total > 0 else 0,
    }


def train_model(num_epochs=15, batch_size=32, learning_rate=5e-5):
    """训练模型"""
    print("="*60)
    print("Audio HMTL V2 训练")
    print("="*60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print(f"\n加载标签: {LABELS_CSV}")
    label_df = pd.read_csv(LABELS_CSV)
    
    print(f"加载缓存: {CACHE_FILE}")
    cached_features = torch.load(CACHE_FILE)
    print(f"缓存样本数: {len(cached_features)}")
    
    # 划分数据集
    train_df, test_df = train_test_split(label_df, test_size=0.2, random_state=42)
    
    train_dataset = CachedAudioDatasetV2(train_df, cached_features)
    test_dataset = CachedAudioDatasetV2(test_df, cached_features)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 模型
    print("\n创建模型...")
    model = AudioHMTLClassifier().to(device)
    
    # 损失函数
    criterion = HMTLAudioLossV2(
        weight_7=1.0,
        weight_4=0.5,
        weight_3=0.3,
        weight_A=0.1,
        weight_V=0.1,
        class_weights_7=CLASS_WEIGHTS_7.to(device),
        class_weights_4=CLASS_WEIGHTS_4.to(device)
    )
    
    # 优化器 - 分层学习率
    backbone_params = list(model.wav2vec2.parameters())
    head_params = (list(model.classifier_7.parameters()) +
                   list(model.classifier_4.parameters()) +
                   list(model.classifier_3.parameters()) +
                   list(model.regressor_A.parameters()) +
                   list(model.regressor_V.parameters()) +
                   list(model.dim_reducer.parameters()))
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate},
        {'params': head_params, 'lr': learning_rate * 10}
    ])
    
    # 训练
    best_acc_7 = 0
    
    print(f"\n开始训练: {num_epochs} epochs")
    print("-"*60)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            if batch is None:
                continue
            
            features = batch['features'].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k != 'features'}
            
            optimizer.zero_grad()
            outputs = model(features)
            loss_dict = criterion(outputs, targets)
            
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_dict['total_loss'].item()
            pbar.set_postfix({'loss': f"{loss_dict['total_loss'].item():.4f}"})
        
        # 评估
        eval_result = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, "
              f"Acc_7={eval_result['acc_7']*100:.2f}%, "
              f"Acc_4={eval_result['acc_4']*100:.2f}%")
        
        # 保存最佳模型
        if eval_result['acc_7'] > best_acc_7:
            best_acc_7 = eval_result['acc_7']
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 保存最佳模型: {best_acc_7*100:.2f}%")
    
    print("\n" + "="*60)
    print(f"训练完成! 最佳 7类准确率: {best_acc_7*100:.2f}%")
    print(f"模型保存到: {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    train_model(num_epochs=15, batch_size=32, learning_rate=5e-5)
