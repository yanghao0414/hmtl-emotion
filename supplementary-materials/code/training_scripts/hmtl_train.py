#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL情绪识别模型训练脚本
支持GPU训练、早停、模型保存等
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "02_模型代码")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import json
import os
from datetime import datetime
from tqdm import tqdm

from hmtl_model import HMTLEmotionModel, HMTLLoss
from hmtl_dataset import create_dataloaders
from hmtl_utils import LABEL_4_NAMES, LABEL_3_NAMES, predict_original_emotion


class HMTLTrainer:
    """HMTL训练器"""
    
    def __init__(self,
                 train_path: str,
                 eval_path: str,
                 model_save_dir: str = str(_PROJECT_ROOT / "06_模型文件" / "hmtl_models"),
                 bert_model_name: str = 'bert-base-chinese',
                 batch_size: int = 16,
                 max_length: int = 128,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 10,
                 warmup_steps: int = 100,
                 device: str = None):
        
        self.train_path = train_path
        self.eval_path = eval_path
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        
        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和数据
        print("\n加载数据...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.train_loader, self.eval_loader = create_dataloaders(
            train_path, eval_path, self.tokenizer, 
            batch_size, max_length
        )
        
        # 创建模型
        print("\n创建模型...")
        self.model = HMTLEmotionModel(bert_model_name).to(self.device)
        
        # 损失函数
        self.criterion = HMTLLoss(
            weight_4=1.0,
            weight_3=0.8,
            weight_arousal=0.5,
            weight_valence=0.5
        )
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_acc_4': [],
            'eval_acc_3': [],
            'eval_acc_7': []  # 推断的7类准确率
        }
        
        self.best_eval_loss = float('inf')
        self.best_eval_acc_4 = 0.0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_loss_4 = 0
        total_loss_3 = 0
        total_loss_arousal = 0
        total_loss_valence = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            label_4 = batch['label_4'].to(self.device)
            label_3 = batch['label_3'].to(self.device)
            arousal = batch['arousal'].to(self.device)
            valence = batch['valence'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            targets = {
                'label_4': label_4,
                'label_3': label_3,
                'arousal': arousal,
                'valence': valence
            }
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # 累计损失
            total_loss += loss.item()
            total_loss_4 += loss_dict['loss_4']
            total_loss_3 += loss_dict['loss_3']
            total_loss_arousal += loss_dict['loss_arousal']
            total_loss_valence += loss_dict['loss_valence']
            
            # 更新进度条
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        return {
            'total_loss': avg_loss,
            'loss_4': total_loss_4 / len(self.train_loader),
            'loss_3': total_loss_3 / len(self.train_loader),
            'loss_arousal': total_loss_arousal / len(self.train_loader),
            'loss_valence': total_loss_valence / len(self.train_loader)
        }
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_4 = 0
        correct_3 = 0
        correct_7 = 0  # 推断的7类准确率
        total_samples = 0
        
        # 用于计算回归指标
        arousal_errors = []
        valence_errors = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label_4 = batch['label_4'].to(self.device)
                label_3 = batch['label_3'].to(self.device)
                arousal = batch['arousal'].to(self.device)
                valence = batch['valence'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                
                # 计算损失
                targets = {
                    'label_4': label_4,
                    'label_3': label_3,
                    'arousal': arousal,
                    'valence': valence
                }
                loss_dict = self.criterion(outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # 计算准确率
                pred_4 = torch.argmax(outputs['label_4_logits'], dim=1)
                pred_3 = torch.argmax(outputs['label_3_logits'], dim=1)
                
                correct_4 += (pred_4 == label_4).sum().item()
                correct_3 += (pred_3 == label_3).sum().item()
                
                # 计算7类准确率（通过推断）
                # 这里需要原始7类标签来验证，暂时用label_4作为近似
                # 实际使用时应该加载原始标签
                
                total_samples += label_4.size(0)
                
                # 回归误差
                arousal_errors.extend(
                    (outputs['arousal'] - arousal).abs().cpu().tolist()
                )
                valence_errors.extend(
                    (outputs['valence'] - valence).abs().cpu().tolist()
                )
        
        avg_loss = total_loss / len(self.eval_loader)
        acc_4 = correct_4 / total_samples
        acc_3 = correct_3 / total_samples
        avg_arousal_mae = sum(arousal_errors) / len(arousal_errors)
        avg_valence_mae = sum(valence_errors) / len(valence_errors)
        
        return {
            'loss': avg_loss,
            'acc_4': acc_4,
            'acc_3': acc_3,
            'arousal_mae': avg_arousal_mae,
            'valence_mae': avg_valence_mae
        }
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始HMTL模型训练")
        print("="*60)
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"评估样本: {len(self.eval_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print("="*60)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # 训练
            train_metrics = self.train_epoch()
            print(f"\n训练损失: {train_metrics['total_loss']:.4f}")
            print(f"  - Loss 4分类: {train_metrics['loss_4']:.4f}")
            print(f"  - Loss 3分类: {train_metrics['loss_3']:.4f}")
            print(f"  - Loss Arousal: {train_metrics['loss_arousal']:.4f}")
            print(f"  - Loss Valence: {train_metrics['loss_valence']:.4f}")
            
            # 评估
            eval_metrics = self.evaluate()
            print(f"\n评估结果:")
            print(f"  - 总损失: {eval_metrics['loss']:.4f}")
            print(f"  - 4分类准确率: {eval_metrics['acc_4']:.2%}")
            print(f"  - 3分类准确率: {eval_metrics['acc_3']:.2%}")
            print(f"  - Arousal MAE: {eval_metrics['arousal_mae']:.4f}")
            print(f"  - Valence MAE: {eval_metrics['valence_mae']:.4f}")
            
            # 保存历史
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['eval_loss'].append(eval_metrics['loss'])
            self.history['eval_acc_4'].append(eval_metrics['acc_4'])
            self.history['eval_acc_3'].append(eval_metrics['acc_3'])
            
            # 保存最佳模型
            if eval_metrics['acc_4'] > self.best_eval_acc_4:
                self.best_eval_acc_4 = eval_metrics['acc_4']
                self.save_model('best_model.pt', eval_metrics)
                print(f"\n✓ 保存最佳模型 (4分类准确率: {eval_metrics['acc_4']:.2%})")
        
        # 保存最终模型
        self.save_model('final_model.pt', eval_metrics)
        self.save_history()
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        print(f"最佳4分类准确率: {self.best_eval_acc_4:.2%}")
        print(f"模型保存在: {self.model_save_dir}")
    
    def save_model(self, filename, metrics):
        """保存模型"""
        save_path = os.path.join(self.model_save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, save_path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.model_save_dir, 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)


def main():
    """主函数"""
    # 数据路径
    train_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json")
    eval_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set_hmtl.json")
    
    # 检查数据是否存在
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print("✗ 请先运行 convert_to_hmtl.py 转换数据集!")
        return
    
    # 创建训练器
    trainer = HMTLTrainer(
        train_path=train_path,
        eval_path=eval_path,
        batch_size=16,
        max_length=128,
        learning_rate=2e-5,
        num_epochs=10,
        warmup_steps=100
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
