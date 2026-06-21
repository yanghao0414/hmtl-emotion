#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频模型训练脚本
使用语音情绪数据集训练多任务情绪识别模型
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = str(PROJECT_ROOT / "05_数据文件" / "labels" / "audio_hmtl_labels_v2.csv")
MODEL_SAVE_PATH = str(PROJECT_ROOT / "06_模型文件" / "audio_hmtl_trained.pt")
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000
MAX_LENGTH = 5 * SAMPLE_RATE  # 5秒

# 4类和7类标签映射
LABEL_4_NAMES = ['积极', '激活消极', '非激活消极', '平静']
LABEL_7_NAMES = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']


def load_audio(audio_path, target_sr=16000, max_length=None):
    """加载音频文件"""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr)
    except ImportError:
        from scipy.io import wavfile
        sr, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / 32768.0
        if sr != target_sr:
            # 简单重采样
            ratio = target_sr / sr
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * ratio)),
                np.arange(len(audio)),
                audio
            )
    
    # 截断或填充
    if max_length:
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
    
    return audio


def extract_features(audio, sr=16000):
    """提取音频特征 (MFCC + 统计特征)"""
    try:
        import librosa
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 其他特征
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # 合并特征
        features = np.concatenate([mfcc_mean, mfcc_std, [zcr, rms]])
        
    except ImportError:
        # 简化版特征
        features = np.zeros(82)
        features[0] = np.mean(audio)
        features[1] = np.std(audio)
        features[2] = np.max(np.abs(audio))
    
    return features.astype(np.float32)


class AudioEmotionDataset(Dataset):
    """音频情绪数据集"""
    
    def __init__(self, df, max_length=MAX_LENGTH, use_cache=True):
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.cache = {}
        self.use_cache = use_cache
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['audio_full_path']
        
        # 尝试从缓存加载
        if self.use_cache and audio_path in self.cache:
            features = self.cache[audio_path]
        else:
            try:
                audio = load_audio(audio_path, max_length=self.max_length)
                features = extract_features(audio)
                if self.use_cache:
                    self.cache[audio_path] = features
            except Exception as e:
                features = np.zeros(82, dtype=np.float32)
        
        # 标签
        label_4 = int(row['label_4_emotion'])
        label_7 = int(row['label_7_emotion'])
        label_3 = int(row['label_3_polarity'])
        arousal = float(row['true_arousal'])
        valence = float(row['true_valence'])
        
        return {
            'features': torch.tensor(features),
            'label_4': label_4,
            'label_7': label_7,
            'label_3': label_3,
            'arousal': arousal,
            'valence': valence
        }


class AudioEmotionModel(nn.Module):
    """音频情绪识别模型"""
    
    def __init__(self, input_dim=82, num_classes_4=4, num_classes_7=7, num_classes_3=3):
        super().__init__()
        
        # 特征提取
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # 多任务头
        self.head_4 = nn.Linear(64, num_classes_4)
        self.head_7 = nn.Linear(64, num_classes_7)
        self.head_3 = nn.Linear(64, num_classes_3)
        self.head_arousal = nn.Linear(64, 1)
        self.head_valence = nn.Linear(64, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        
        out_4 = self.head_4(features)
        out_7 = self.head_7(features)
        out_3 = self.head_3(features)
        arousal = torch.tanh(self.head_arousal(features))
        valence = torch.tanh(self.head_valence(features))
        
        return out_4, out_7, out_3, arousal, valence


def load_data():
    """加载数据"""
    print("📂 加载数据...")
    
    df = pd.read_csv(LABELS_FILE)
    print(f"  总样本数: {len(df)}")
    
    # 检查文件是否存在
    valid_mask = df['audio_full_path'].apply(os.path.exists)
    df = df[valid_mask]
    print(f"  有效样本数: {len(df)}")
    
    if len(df) == 0:
        print("❌ 没有找到有效的音频文件！")
        print("  请检查路径是否正确")
        return None, None
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_4_emotion'])
    print(f"  训练集: {len(train_df)}, 验证集: {len(val_df)}")
    
    return train_df, val_df


def train_epoch(model, loader, optimizer, criterion_ce, criterion_mse):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct_4 = 0
    total = 0
    
    for batch in tqdm(loader, desc="训练中", leave=False):
        features = batch['features'].to(DEVICE)
        label_4 = batch['label_4'].to(DEVICE)
        label_7 = batch['label_7'].to(DEVICE)
        label_3 = batch['label_3'].to(DEVICE)
        arousal = batch['arousal'].float().to(DEVICE)
        valence = batch['valence'].float().to(DEVICE)
        
        optimizer.zero_grad()
        
        out_4, out_7, out_3, pred_arousal, pred_valence = model(features)
        
        # 计算损失
        loss_4 = criterion_ce(out_4, label_4)
        loss_7 = criterion_ce(out_7, label_7)
        loss_3 = criterion_ce(out_3, label_3)
        loss_arousal = criterion_mse(pred_arousal.squeeze(), arousal)
        loss_valence = criterion_mse(pred_valence.squeeze(), valence)
        
        # 总损失
        loss = loss_4 * 2.0 + loss_7 * 1.0 + loss_3 * 0.5 + loss_arousal * 0.5 + loss_valence * 0.5
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = out_4.max(1)
        total += label_4.size(0)
        correct_4 += predicted.eq(label_4).sum().item()
    
    return total_loss / len(loader), correct_4 / total


def validate(model, loader, criterion_ce, criterion_mse):
    """验证"""
    model.eval()
    total_loss = 0
    correct_4 = 0
    correct_7 = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="验证中", leave=False):
            features = batch['features'].to(DEVICE)
            label_4 = batch['label_4'].to(DEVICE)
            label_7 = batch['label_7'].to(DEVICE)
            label_3 = batch['label_3'].to(DEVICE)
            arousal = batch['arousal'].float().to(DEVICE)
            valence = batch['valence'].float().to(DEVICE)
            
            out_4, out_7, out_3, pred_arousal, pred_valence = model(features)
            
            loss_4 = criterion_ce(out_4, label_4)
            loss_7 = criterion_ce(out_7, label_7)
            loss_3 = criterion_ce(out_3, label_3)
            loss_arousal = criterion_mse(pred_arousal.squeeze(), arousal)
            loss_valence = criterion_mse(pred_valence.squeeze(), valence)
            
            loss = loss_4 * 2.0 + loss_7 * 1.0 + loss_3 * 0.5 + loss_arousal * 0.5 + loss_valence * 0.5
            
            total_loss += loss.item()
            _, predicted_4 = out_4.max(1)
            _, predicted_7 = out_7.max(1)
            total += label_4.size(0)
            correct_4 += predicted_4.eq(label_4).sum().item()
            correct_7 += predicted_7.eq(label_7).sum().item()
    
    return total_loss / len(loader), correct_4 / total, correct_7 / total


def main():
    print("="*60)
    print("🎵 音频模型训练")
    print(f"   设备: {DEVICE}")
    print("="*60)
    
    # 加载数据
    train_df, val_df = load_data()
    
    if train_df is None:
        return
    
    # 创建数据加载器
    print("\n📦 创建数据加载器...")
    train_dataset = AudioEmotionDataset(train_df)
    val_dataset = AudioEmotionDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = AudioEmotionModel().to(DEVICE)
    
    # 损失函数和优化器
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # 训练
    print("\n🚀 开始训练...")
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion_ce, criterion_mse)
        val_loss, val_acc_4, val_acc_7 = validate(model, val_loader, criterion_ce, criterion_mse)
        
        scheduler.step(val_acc_4)
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2%} | "
              f"Val Loss={val_loss:.4f}, Val 4类={val_acc_4:.2%}, Val 7类={val_acc_7:.2%}")
        
        # 保存最佳模型
        if val_acc_4 > best_acc:
            best_acc = val_acc_4
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy_4': val_acc_4,
                'accuracy_7': val_acc_7,
                'epoch': epoch
            }, MODEL_SAVE_PATH)
            print(f"  ✅ 保存最佳模型 (4类准确率: {val_acc_4:.2%})")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  ⚠️ 早停: 10个epoch没有提升")
                break
    
    print("\n" + "="*60)
    print(f"✅ 训练完成！最佳4类准确率: {best_acc:.2%}")
    print(f"   模型保存至: {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
