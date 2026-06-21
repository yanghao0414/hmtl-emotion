#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL情绪识别模型训练脚本 V2
整合三大优化策略
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import json
import os
from datetime import datetime
from tqdm import tqdm

from hmtl_model_v2 import HMTLEmotionModelV2, HMTLLossV2
from hmtl_dataset import HMTLEmotionDataset
from torch.utils.data import DataLoader
from hmtl_utils import LABEL_4_NAMES, LABEL_3_NAMES


# 7类情绪映射（包含细粒度情绪映射）
EMOTION_7_MAP = {
    # 主要7类
    '愤怒': 0, '焦虑': 1, '快乐': 2, '悲伤': 3, 
    '失望': 4, '支持': 5, '平静': 6,
    
    # 细粒度→愤怒
    '生气': 0,
    
    # 细粒度→焦虑
    '紧张': 1, '担心': 1, '害怕': 1, '恐惧': 1, '困惑': 1, '犹豫': 1,
    
    # 细粒度→快乐
    '兴奋': 2, '激动': 2, '希望': 2, '期待': 2, '自信': 2,
    
    # 细粒度→悲伤
    '沮丧': 3, '无助': 3,
    
    # 细粒度→支持
    '理解': 5, '安慰': 5, '鼓励': 5,
    
    # 细粒度→平静
    '放松': 6, '坚定': 6
}

EMOTION_7_NAMES = {v: k for k, v in list(EMOTION_7_MAP.items())[:7]}


class HMTLDatasetV2(HMTLEmotionDataset):
    """扩展数据集以支持7分类标签"""
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # BERT编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取7类情绪标签
        emotion_7_name = item.get('original_emotion', '平静')
        label_7 = EMOTION_7_MAP.get(emotion_7_name, 6)  # 默认为平静
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_4': torch.tensor(item['label_4'], dtype=torch.long),
            'label_3': torch.tensor(item['label_3'], dtype=torch.long),
            'label_7': torch.tensor(label_7, dtype=torch.long),  # 新增
            'arousal': torch.tensor(item['arousal'], dtype=torch.float),
            'valence': torch.tensor(item['valence'], dtype=torch.float)
        }


def create_dataloaders_v2(train_path, eval_path, tokenizer, batch_size=16, max_length=128):
    """创建V2数据加载器"""
    train_dataset = HMTLDatasetV2(train_path, tokenizer, max_length)
    eval_dataset = HMTLDatasetV2(eval_path, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=0, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, eval_loader


class HMTLTrainerV2:
    """HMTL训练器 V2"""
    
    def __init__(self,
                 train_path: str,
                 eval_path: str,
                 model_save_dir: str = str(Path(__file__).resolve().parents[3] / "06_模型文件" / "hmtl_models_v2"),
                 bert_model_name: str = 'bert-base-chinese',
                 batch_size: int = 16,
                 max_length: int = 128,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 15,
                 warmup_steps: int = 100,
                 # V2参数
                 use_focal_loss: bool = True,
                 focal_gamma: float = 2.0,
                 weight_arousal: float = 0.8,  # 提高Arousal权重
                 device: str = None):
        
        self.train_path = train_path
        self.eval_path = eval_path
        self.model_save_dir = model_save_dir
        self.num_epochs = num_epochs
        
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载数据
        print("\n加载数据...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.train_loader, self.eval_loader = create_dataloaders_v2(
            train_path, eval_path, self.tokenizer, batch_size, max_length
        )
        
        # 创建模型V2
        print("\n创建模型 V2...")
        self.model = HMTLEmotionModelV2(bert_model_name, num_emotions=7).to(self.device)
        
        # 损失函数V2（包含优化策略）
        print(f"使用Focal Loss: {use_focal_loss}")
        print(f"Arousal权重: {weight_arousal}")
        
        self.criterion = HMTLLossV2(
            weight_4=1.0,
            weight_3=0.8,
            weight_7=1.2,  # 7分类权重
            weight_arousal=weight_arousal,  # 提高权重解决愤怒↔焦虑
            weight_valence=0.5,
            use_focal_loss=use_focal_loss,  # 使用Focal Loss解决失望↔悲伤
            focal_gamma=focal_gamma
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
            'eval_acc_7': []  # 新增
        }
        
        self.best_eval_acc_7 = 0.0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            label_4 = batch['label_4'].to(self.device)
            label_3 = batch['label_3'].to(self.device)
            label_7 = batch['label_7'].to(self.device)
            arousal = batch['arousal'].to(self.device)
            valence = batch['valence'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            
            # 计算损失
            targets = {
                'label_4': label_4,
                'label_3': label_3,
                'label_7': label_7,
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
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_4 = 0
        correct_3 = 0
        correct_7 = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label_4 = batch['label_4'].to(self.device)
                label_3 = batch['label_3'].to(self.device)
                label_7 = batch['label_7'].to(self.device)
                arousal = batch['arousal'].to(self.device)
                valence = batch['valence'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                targets = {
                    'label_4': label_4,
                    'label_3': label_3,
                    'label_7': label_7,
                    'arousal': arousal,
                    'valence': valence
                }
                loss_dict = self.criterion(outputs, targets)
                total_loss += loss_dict['total_loss'].item()
                
                # 准确率
                pred_4 = torch.argmax(outputs['label_4_logits'], dim=1)
                pred_3 = torch.argmax(outputs['label_3_logits'], dim=1)
                pred_7 = torch.argmax(outputs['label_7_logits'], dim=1)
                
                correct_4 += (pred_4 == label_4).sum().item()
                correct_3 += (pred_3 == label_3).sum().item()
                correct_7 += (pred_7 == label_7).sum().item()
                
                total_samples += label_4.size(0)
        
        return {
            'loss': total_loss / len(self.eval_loader),
            'acc_4': correct_4 / total_samples,
            'acc_3': correct_3 / total_samples,
            'acc_7': correct_7 / total_samples
        }
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("开始HMTL V2 模型训练")
        print("="*60)
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"评估样本: {len(self.eval_loader.dataset)}")
        print(f"优化策略:")
        print(f"  - Focal Loss解决失望↔悲伤")
        print(f"  - Arousal权重0.8解决愤怒↔焦虑")
        print(f"  - 数据增强解决支持↔快乐（需先运行data_augmentation.py）")
        print("="*60)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # 训练
            train_loss = self.train_epoch()
            print(f"\n训练损失: {train_loss:.4f}")
            
            # 评估
            eval_metrics = self.evaluate()
            print(f"\n评估结果:")
            print(f"  - 总损失: {eval_metrics['loss']:.4f}")
            print(f"  - 4分类准确率: {eval_metrics['acc_4']:.2%}")
            print(f"  - 3分类准确率: {eval_metrics['acc_3']:.2%}")
            print(f"  - 7分类准确率: {eval_metrics['acc_7']:.2%} ⭐")
            
            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['eval_loss'].append(eval_metrics['loss'])
            self.history['eval_acc_4'].append(eval_metrics['acc_4'])
            self.history['eval_acc_3'].append(eval_metrics['acc_3'])
            self.history['eval_acc_7'].append(eval_metrics['acc_7'])
            
            # 保存最佳模型（以7分类为准）
            if eval_metrics['acc_7'] > self.best_eval_acc_7:
                self.best_eval_acc_7 = eval_metrics['acc_7']
                self.save_model('best_model_v2.pt', eval_metrics)
                print(f"\n✓ 保存最佳模型 (7分类准确率: {eval_metrics['acc_7']:.2%})")
        
        # 保存最终模型
        self.save_model('final_model_v2.pt', eval_metrics)
        self.save_history()
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        print(f"最佳7分类准确率: {self.best_eval_acc_7:.2%}")
    
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
        history_path = os.path.join(self.model_save_dir, 'training_history_v2.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)


def main():
    """主函数"""
    # 数据路径
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    train_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl_augmented.json")
    eval_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set_hmtl.json")
    
    # 检查增强数据是否存在
    if not os.path.exists(train_path):
        print("⚠️ 未找到增强数据，使用原始数据")
        print("  建议先运行: python data_augmentation.py")
        train_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json")
    
    # 创建训练器
    trainer = HMTLTrainerV2(
        train_path=train_path,
        eval_path=eval_path,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=15,  # 更多epochs
        use_focal_loss=True,  # 使用Focal Loss
        focal_gamma=2.0,
        weight_arousal=0.8  # 提高Arousal权重
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
