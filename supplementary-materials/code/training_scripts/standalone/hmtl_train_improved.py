#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL模型改进训练脚本
解决过拟合问题:
1. 数据增强
2. 增强正则化
3. 早停策略
4. 交叉验证
"""

import sys
import os
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer, BertModel
import json
import random
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from collections import Counter

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class TextAugmenter:
    """文本数据增强器"""
    
    def __init__(self):
        # 同义词词典 (简化版)
        self.synonyms = {
            '开心': ['高兴', '快乐', '愉快', '欢喜'],
            '难过': ['伤心', '悲伤', '难受', '痛苦'],
            '生气': ['愤怒', '恼火', '气愤', '发怒'],
            '担心': ['焦虑', '忧虑', '担忧', '不安'],
            '失望': ['沮丧', '灰心', '泄气'],
            '平静': ['安静', '平和', '淡定', '从容'],
            '很': ['非常', '十分', '特别', '极其'],
            '好': ['棒', '不错', '优秀', '出色'],
        }
    
    def synonym_replacement(self, text, n=1):
        """同义词替换"""
        words = list(text)
        augmented_texts = [text]
        
        for word, syns in self.synonyms.items():
            if word in text:
                for syn in random.sample(syns, min(n, len(syns))):
                    new_text = text.replace(word, syn, 1)
                    if new_text != text:
                        augmented_texts.append(new_text)
        
        return augmented_texts
    
    def random_deletion(self, text, p=0.1):
        """随机删除字符"""
        if len(text) <= 5:
            return text
        
        chars = list(text)
        new_chars = [c for c in chars if random.random() > p]
        
        if len(new_chars) < 3:
            return text
        
        return ''.join(new_chars)
    
    def random_swap(self, text, n=1):
        """随机交换字符位置"""
        chars = list(text)
        for _ in range(n):
            if len(chars) < 2:
                break
            i, j = random.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)
    
    def augment(self, text, num_aug=2):
        """综合数据增强"""
        augmented = [text]
        
        # 同义词替换
        augmented.extend(self.synonym_replacement(text)[:num_aug])
        
        # 随机删除
        if random.random() < 0.3:
            augmented.append(self.random_deletion(text))
        
        # 随机交换
        if random.random() < 0.2:
            augmented.append(self.random_swap(text))
        
        return list(set(augmented))[:num_aug + 1]


class HMTLDatasetImproved(Dataset):
    """改进的HMTL数据集"""
    
    def __init__(self, data, tokenizer, max_length=128, augment=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augmenter = TextAugmenter() if augment else None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 数据增强
        if self.augment and self.augmenter and random.random() < 0.5:
            augmented_texts = self.augmenter.augment(text, num_aug=1)
            text = random.choice(augmented_texts)
        
        # 编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_4': torch.tensor(item['label_4'], dtype=torch.long),
            'label_3': torch.tensor(item['label_3'], dtype=torch.long),
            'label_7': torch.tensor(item.get('label_7', 0), dtype=torch.long),
            'arousal': torch.tensor(item['arousal'], dtype=torch.float),
            'valence': torch.tensor(item['valence'], dtype=torch.float)
        }


class HMTLModelImproved(nn.Module):
    """改进的HMTL模型 - 增强正则化"""
    
    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.5):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        
        # 增强的dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4分类头 - 增加正则化
        self.classifier_4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 4)
        )
        
        # 3分类头
        self.classifier_3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(128, 3)
        )
        
        # 7分类头
        self.classifier_7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(256, 7)
        )
        
        # Arousal回归头
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Valence回归头
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        label_4_logits = self.classifier_4(cls_output)
        label_3_logits = self.classifier_3(cls_output)
        label_7_logits = self.classifier_7(cls_output)
        arousal = self.arousal_head(cls_output).squeeze(-1)
        valence = self.valence_head(cls_output).squeeze(-1)
        
        return {
            'label_4_logits': label_4_logits,
            'label_3_logits': label_3_logits,
            'label_7_logits': label_7_logits,
            'arousal': arousal,
            'valence': valence
        }


class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失 - 减少过拟合"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)
        
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smooth_target = one_hot * confidence + (1 - one_hot) * smooth_value
        
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(smooth_target * log_probs).sum(dim=1).mean()
        
        return loss


def train_epoch(model, dataloader, optimizer, criterion_4, criterion_3, criterion_7, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct_4 = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label_4 = batch['label_4'].to(device)
        label_3 = batch['label_3'].to(device)
        label_7 = batch['label_7'].to(device)
        arousal = batch['arousal'].to(device)
        valence = batch['valence'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        
        # 计算损失
        loss_4 = criterion_4(outputs['label_4_logits'], label_4)
        loss_3 = criterion_3(outputs['label_3_logits'], label_3)
        loss_7 = criterion_7(outputs['label_7_logits'], label_7)
        loss_arousal = F.mse_loss(outputs['arousal'], arousal)
        loss_valence = F.mse_loss(outputs['valence'], valence)
        
        loss = loss_4 + 0.8 * loss_3 + 1.2 * loss_7 + 0.5 * loss_arousal + 0.5 * loss_valence
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        pred_4 = torch.argmax(outputs['label_4_logits'], dim=1)
        correct_4 += (pred_4 == label_4).sum().item()
        total += label_4.size(0)
    
    return total_loss / len(dataloader), correct_4 / total


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    correct_4 = 0
    correct_7 = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_4 = batch['label_4'].to(device)
            label_7 = batch['label_7'].to(device)
            
            outputs = model(input_ids, attention_mask)
            
            pred_4 = torch.argmax(outputs['label_4_logits'], dim=1)
            pred_7 = torch.argmax(outputs['label_7_logits'], dim=1)
            
            correct_4 += (pred_4 == label_4).sum().item()
            correct_7 += (pred_7 == label_7).sum().item()
            total += label_4.size(0)
    
    return correct_4 / total, correct_7 / total


def train_with_cross_validation(data_path, output_dir, n_splits=5, epochs=20):
    """使用交叉验证训练"""
    print("🚀 开始交叉验证训练")
    print("="*60)
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 数据集大小: {len(data)}")
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # K折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f"\n📁 Fold {fold + 1}/{n_splits}")
        print("-"*40)
        
        # 划分数据
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        # 创建数据集 (训练集使用数据增强)
        train_dataset = HMTLDatasetImproved(train_data, tokenizer, augment=True)
        val_dataset = HMTLDatasetImproved(val_data, tokenizer, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # 创建模型
        model = HMTLModelImproved(dropout=0.5).to(device)
        
        # 优化器 (使用weight decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # 损失函数 (使用标签平滑)
        criterion_4 = LabelSmoothingLoss(num_classes=4, smoothing=0.1)
        criterion_3 = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
        criterion_7 = LabelSmoothingLoss(num_classes=7, smoothing=0.1)
        
        # 早停
        early_stopping = EarlyStopping(patience=3, mode='max')
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, 
                criterion_4, criterion_3, criterion_7, device
            )
            
            val_acc_4, val_acc_7 = evaluate(model, val_loader, device)
            
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc_4={val_acc_4:.4f}, val_acc_7={val_acc_7:.4f}")
            
            if val_acc_4 > best_val_acc:
                best_val_acc = val_acc_4
            
            # 早停检查
            if early_stopping(val_acc_4, model):
                print(f"  ⏹️ 早停于 epoch {epoch+1}")
                break
        
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc_4': best_val_acc,
            'final_val_acc_4': val_acc_4,
            'final_val_acc_7': val_acc_7
        })
        
        print(f"  ✅ Fold {fold+1} 最佳验证准确率: {best_val_acc:.4f}")
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 交叉验证结果汇总:")
    avg_acc = np.mean([r['best_val_acc_4'] for r in fold_results])
    std_acc = np.std([r['best_val_acc_4'] for r in fold_results])
    
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_acc_4']:.4f}")
    
    print(f"\n  平均准确率: {avg_acc:.4f} ± {std_acc:.4f}")
    print("="*60)
    
    return fold_results


def main():
    """主函数"""
    print("🧪 HMTL模型改进训练")
    print("="*60)
    
    # 检查数据文件
    data_candidates = [
        PROJECT_ROOT / "05_数据集" / "hmtl_train.json",
        PROJECT_ROOT / "05_数据文件" / "hmtl_train.json",
    ]

    data_path = None
    for p in data_candidates:
        if p.exists():
            data_path = str(p)
            break
    
    if not data_path:
        print("❌ 数据文件不存在:")
        for p in data_candidates:
            print(f"- {p}")
        print("请确保训练数据文件存在")
        return
    
    output_dir = str(PROJECT_ROOT / "06_模型文件" / "hmtl_models_improved")
    os.makedirs(output_dir, exist_ok=True)
    
    # 交叉验证训练
    results = train_with_cross_validation(
        data_path=data_path,
        output_dir=output_dir,
        n_splits=5,
        epochs=15
    )
    
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main()
