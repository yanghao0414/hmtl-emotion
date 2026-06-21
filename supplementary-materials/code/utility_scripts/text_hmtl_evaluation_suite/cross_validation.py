#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text HMTL 模型 5-fold 交叉验证脚本
用于获得更可靠的性能估计

使用方法:
    python cross_validation.py --data all_data.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '02_模型代码'))

from transformers import BertTokenizer, BertModel
from hmtl_model_v2 import HMTLEmotionModelV2, HMTLLossV2

EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}


class TextEmotionDataset(Dataset):
    """文本情绪数据集"""
    
    def __init__(self, texts, labels_7, labels_4, labels_3, arousals, valences, tokenizer, max_length=128):
        self.texts = texts
        self.labels_7 = labels_7
        self.labels_4 = labels_4
        self.labels_3 = labels_3
        self.arousals = arousals
        self.valences = valences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_7': torch.tensor(self.labels_7[idx], dtype=torch.long),
            'label_4': torch.tensor(self.labels_4[idx], dtype=torch.long),
            'label_3': torch.tensor(self.labels_3[idx], dtype=torch.long),
            'arousal': torch.tensor(self.arousals[idx], dtype=torch.float),
            'valence': torch.tensor(self.valences[idx], dtype=torch.float),
        }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = {
            'label_7': batch['label_7'].to(device),
            'label_4': batch['label_4'].to(device),
            'label_3': batch['label_3'].to(device),
            'arousal': batch['arousal'].to(device),
            'valence': batch['valence'].to(device),
        }
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss_dict = criterion(outputs, targets)
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss_dict['total_loss'].item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_pred_7 = []
    all_true_7 = []
    all_pred_arousal = []
    all_true_arousal = []
    all_pred_valence = []
    all_true_valence = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            pred_7 = outputs['label_7_logits'].argmax(dim=1).cpu().numpy()
            all_pred_7.extend(pred_7)
            all_true_7.extend(batch['label_7'].numpy())
            
            all_pred_arousal.extend(outputs['arousal'].cpu().numpy())
            all_true_arousal.extend(batch['arousal'].numpy())
            
            all_pred_valence.extend(outputs['valence'].cpu().numpy())
            all_true_valence.extend(batch['valence'].numpy())
    
    acc_7 = accuracy_score(all_true_7, all_pred_7)
    f1_7 = f1_score(all_true_7, all_pred_7, average='macro', zero_division=0)
    mae_arousal = mean_absolute_error(all_true_arousal, all_pred_arousal)
    mae_valence = mean_absolute_error(all_true_valence, all_pred_valence)
    
    return {
        'accuracy_7': acc_7,
        'macro_f1_7': f1_7,
        'mae_arousal': mae_arousal,
        'mae_valence': mae_valence
    }


def run_cross_validation(df: pd.DataFrame, n_folds: int = 5, epochs: int = 10):
    """运行交叉验证"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 准备数据
    texts = df['text'].tolist()
    labels_7 = df['label_7'].astype(int).tolist()
    labels_4 = df['label_4'].astype(int).tolist() if 'label_4' in df.columns else [0] * len(df)
    labels_3 = df['label_3'].astype(int).tolist() if 'label_3' in df.columns else [0] * len(df)
    arousals = df['arousal'].astype(float).tolist()
    valences = df['valence'].astype(float).tolist()
    
    # 分层K折
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels_7)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        # 划分数据
        train_texts = [texts[i] for i in train_idx]
        train_labels_7 = [labels_7[i] for i in train_idx]
        train_labels_4 = [labels_4[i] for i in train_idx]
        train_labels_3 = [labels_3[i] for i in train_idx]
        train_arousals = [arousals[i] for i in train_idx]
        train_valences = [valences[i] for i in train_idx]
        
        val_texts = [texts[i] for i in val_idx]
        val_labels_7 = [labels_7[i] for i in val_idx]
        val_labels_4 = [labels_4[i] for i in val_idx]
        val_labels_3 = [labels_3[i] for i in val_idx]
        val_arousals = [arousals[i] for i in val_idx]
        val_valences = [valences[i] for i in val_idx]
        
        print(f"训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
        
        # 创建数据集
        train_dataset = TextEmotionDataset(
            train_texts, train_labels_7, train_labels_4, train_labels_3,
            train_arousals, train_valences, tokenizer
        )
        val_dataset = TextEmotionDataset(
            val_texts, val_labels_7, val_labels_4, val_labels_3,
            val_arousals, val_valences, tokenizer
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 创建模型
        model = HMTLEmotionModelV2().to(device)
        criterion = HMTLLossV2(use_focal_loss=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # 训练
        best_acc = 0
        best_result = None
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_result = evaluate(model, val_loader, device)
            
            if val_result['accuracy_7'] > best_acc:
                best_acc = val_result['accuracy_7']
                best_result = val_result.copy()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
                  f"Acc={val_result['accuracy_7']*100:.2f}%, "
                  f"F1={val_result['macro_f1_7']*100:.2f}%")
        
        fold_results.append(best_result)
        print(f"\nFold {fold+1} 最佳结果: Acc={best_acc*100:.2f}%")
    
    # 汇总结果
    print("\n" + "="*60)
    print("交叉验证汇总")
    print("="*60)
    
    accs = [r['accuracy_7'] for r in fold_results]
    f1s = [r['macro_f1_7'] for r in fold_results]
    mae_as = [r['mae_arousal'] for r in fold_results]
    mae_vs = [r['mae_valence'] for r in fold_results]
    
    summary = {
        'n_folds': n_folds,
        'accuracy_7': {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'per_fold': accs
        },
        'macro_f1_7': {
            'mean': np.mean(f1s),
            'std': np.std(f1s),
            'per_fold': f1s
        },
        'mae_arousal': {
            'mean': np.mean(mae_as),
            'std': np.std(mae_as)
        },
        'mae_valence': {
            'mean': np.mean(mae_vs),
            'std': np.std(mae_vs)
        }
    }
    
    print(f"7类准确率: {summary['accuracy_7']['mean']*100:.2f}% ± {summary['accuracy_7']['std']*100:.2f}%")
    print(f"Macro-F1: {summary['macro_f1_7']['mean']*100:.2f}% ± {summary['macro_f1_7']['std']*100:.2f}%")
    print(f"Arousal MAE: {summary['mae_arousal']['mean']:.4f} ± {summary['mae_arousal']['std']:.4f}")
    print(f"Valence MAE: {summary['mae_valence']['mean']:.4f} ± {summary['mae_valence']['std']:.4f}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Text HMTL 5-fold 交叉验证')
    parser.add_argument('--data', type=str, required=True, help='数据CSV路径')
    parser.add_argument('--folds', type=int, default=5, help='折数')
    parser.add_argument('--epochs', type=int, default=10, help='每折训练轮数')
    parser.add_argument('--output', type=str, default='cv_results.json', help='输出路径')
    args = parser.parse_args()
    
    # 加载数据
    print("加载数据...")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=['text', 'label_7', 'arousal', 'valence'])
    print(f"有效样本: {len(df)}")
    
    # 运行交叉验证
    summary = run_cross_validation(df, n_folds=args.folds, epochs=args.epochs)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
