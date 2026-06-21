#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉模型训练脚本
使用面部表情数据集训练多任务情绪识别模型
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = str(PROJECT_ROOT / "05_数据文件" / "labels" / "visual_hmtl_labels.csv")
MODEL_SAVE_PATH = str(PROJECT_ROOT / "06_模型文件" / "visual_hmtl_trained.pt")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4类和7类标签映射
LABEL_4_NAMES = ['积极', '激活消极', '非激活消极', '平静']
LABEL_7_NAMES = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']


class FaceEmotionDataset(Dataset):
    """面部表情数据集"""
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图像
        img_path = row['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果加载失败，返回随机图像
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        # 标签
        label_4 = int(row['label_4'])
        label_7 = int(row['label_7'])
        label_3 = int(row['label_3'])
        arousal = float(row['arousal'])
        valence = float(row['valence'])
        
        return {
            'image': image,
            'label_4': label_4,
            'label_7': label_7,
            'label_3': label_3,
            'arousal': arousal,
            'valence': valence
        }


class VisualEmotionModel(nn.Module):
    """视觉情绪识别模型"""
    
    def __init__(self, num_classes_4=4, num_classes_7=7, num_classes_3=3):
        super().__init__()
        
        # 使用预训练的ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结前几层
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        # 获取特征维度
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 多任务头
        self.head_4 = nn.Linear(128, num_classes_4)
        self.head_7 = nn.Linear(128, num_classes_7)
        self.head_3 = nn.Linear(128, num_classes_3)
        self.head_arousal = nn.Linear(128, 1)
        self.head_valence = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        
        out_4 = self.head_4(shared)
        out_7 = self.head_7(shared)
        out_3 = self.head_3(shared)
        arousal = torch.tanh(self.head_arousal(shared))
        valence = torch.tanh(self.head_valence(shared))
        
        return out_4, out_7, out_3, arousal, valence


def load_data():
    """加载数据"""
    print("📂 加载数据...")
    
    df = pd.read_csv(LABELS_FILE)
    print(f"  总样本数: {len(df)}")
    
    # 检查文件是否存在
    valid_mask = df['image_path'].apply(os.path.exists)
    df = df[valid_mask]
    print(f"  有效样本数: {len(df)}")
    
    # 筛选高置信度样本
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= 0.7]
        print(f"  高置信度样本: {len(df)}")
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_4'])
    print(f"  训练集: {len(train_df)}, 验证集: {len(val_df)}")
    
    return train_df, val_df


def create_dataloaders(train_df, val_df):
    """创建数据加载器"""
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FaceEmotionDataset(train_df, transform=train_transform)
    val_dataset = FaceEmotionDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, criterion_ce, criterion_mse):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct_4 = 0
    total = 0
    
    for batch in tqdm(loader, desc="训练中", leave=False):
        images = batch['image'].to(DEVICE)
        label_4 = batch['label_4'].to(DEVICE)
        label_7 = batch['label_7'].to(DEVICE)
        label_3 = batch['label_3'].to(DEVICE)
        arousal = batch['arousal'].float().to(DEVICE)
        valence = batch['valence'].float().to(DEVICE)
        
        optimizer.zero_grad()
        
        out_4, out_7, out_3, pred_arousal, pred_valence = model(images)
        
        # 计算损失
        loss_4 = criterion_ce(out_4, label_4)
        loss_7 = criterion_ce(out_7, label_7)
        loss_3 = criterion_ce(out_3, label_3)
        loss_arousal = criterion_mse(pred_arousal.squeeze(), arousal)
        loss_valence = criterion_mse(pred_valence.squeeze(), valence)
        
        # 总损失 (4类权重最高)
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
            images = batch['image'].to(DEVICE)
            label_4 = batch['label_4'].to(DEVICE)
            label_7 = batch['label_7'].to(DEVICE)
            label_3 = batch['label_3'].to(DEVICE)
            arousal = batch['arousal'].float().to(DEVICE)
            valence = batch['valence'].float().to(DEVICE)
            
            out_4, out_7, out_3, pred_arousal, pred_valence = model(images)
            
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
    print("🎯 视觉模型训练")
    print(f"   设备: {DEVICE}")
    print("="*60)
    
    # 加载数据
    train_df, val_df = load_data()
    train_loader, val_loader = create_dataloaders(train_df, val_df)
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = VisualEmotionModel().to(DEVICE)
    
    # 损失函数和优化器
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 训练
    print("\n🚀 开始训练...")
    best_acc = 0
    
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy_4': val_acc_4,
                'accuracy_7': val_acc_7,
                'epoch': epoch
            }, MODEL_SAVE_PATH)
            print(f"  ✅ 保存最佳模型 (4类准确率: {val_acc_4:.2%})")
    
    print("\n" + "="*60)
    print(f"✅ 训练完成！最佳4类准确率: {best_acc:.2%}")
    print(f"   模型保存至: {MODEL_SAVE_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
