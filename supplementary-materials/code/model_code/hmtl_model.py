#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL情绪识别模型
基于BERT的多任务学习模型
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class HMTLEmotionModel(nn.Module):
    """
    HMTL多任务情绪识别模型
    
    同时预测4个任务:
    - Task 1: 4类情绪分类 (积极/激活消极/非激活消极/平静)
    - Task 2: 3类情感极性 (积极/消极/平静)
    - Task 3: Arousal 唤醒度 (连续值 0-1)
    - Task 4: Valence 效价 (连续值 -1 to 1)
    """
    
    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.3):
        super().__init__()
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768
        
        # 任务1: 4类情绪分类
        self.classifier_4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, 4)
        )
        
        # 任务2: 3类情感极性
        self.classifier_3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 3)
        )
        
        # 任务3: Arousal唤醒度回归
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出0-1
        )
        
        # 任务4: Valence效价回归
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # 输出-1到1
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            dict: {
                'label_4_logits': [batch_size, 4],
                'label_3_logits': [batch_size, 3],
                'arousal': [batch_size],
                'valence': [batch_size]
            }
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 多任务预测
        label_4_logits = self.classifier_4(cls_output)  # [batch_size, 4]
        label_3_logits = self.classifier_3(cls_output)  # [batch_size, 3]
        arousal = self.arousal_head(cls_output).squeeze(-1)  # [batch_size]
        valence = self.valence_head(cls_output).squeeze(-1)  # [batch_size]
        
        return {
            'label_4_logits': label_4_logits,
            'label_3_logits': label_3_logits,
            'arousal': arousal,
            'valence': valence
        }


class HMTLLoss(nn.Module):
    """
    HMTL多任务损失函数
    
    加权组合分类损失和回归损失
    """
    
    def __init__(self, 
                 weight_4=1.0,         # 4类分类权重
                 weight_3=0.8,         # 3类分类权重
                 weight_arousal=0.5,   # Arousal回归权重
                 weight_valence=0.5,   # Valence回归权重
                 class_weights_4=None, # 4类类别权重
                 class_weights_3=None):# 3类类别权重
        super().__init__()
        self.weight_4 = weight_4
        self.weight_3 = weight_3
        self.weight_arousal = weight_arousal
        self.weight_valence = weight_valence
        
        # 分类损失
        self.ce_loss_4 = nn.CrossEntropyLoss(weight=class_weights_4)
        self.ce_loss_3 = nn.CrossEntropyLoss(weight=class_weights_3)
        
        # 回归损失
        self.mse_loss = nn.MSELoss()
    
    def forward(self, outputs, targets):
        """
        计算多任务总损失
        
        Args:
            outputs: 模型输出dict
            targets: 目标标签dict {
                'label_4': [batch_size],
                'label_3': [batch_size],
                'arousal': [batch_size],
                'valence': [batch_size]
            }
        
        Returns:
            dict: {
                'total_loss': tensor,
                'loss_4': float,
                'loss_3': float,
                'loss_arousal': float,
                'loss_valence': float
            }
        """
        # 分类损失
        loss_4 = self.ce_loss_4(outputs['label_4_logits'], targets['label_4'])
        loss_3 = self.ce_loss_3(outputs['label_3_logits'], targets['label_3'])
        
        # 回归损失
        loss_arousal = self.mse_loss(outputs['arousal'], targets['arousal'])
        loss_valence = self.mse_loss(outputs['valence'], targets['valence'])
        
        # 加权总损失
        total_loss = (self.weight_4 * loss_4 + 
                     self.weight_3 * loss_3 + 
                     self.weight_arousal * loss_arousal + 
                     self.weight_valence * loss_valence)
        
        return {
            'total_loss': total_loss,
            'loss_4': loss_4.item(),
            'loss_3': loss_3.item(),
            'loss_arousal': loss_arousal.item(),
            'loss_valence': loss_valence.item()
        }


def test_model():
    """测试模型"""
    print("测试HMTL模型...")
    
    # 创建模型
    model = HMTLEmotionModel()
    model.eval()
    
    # 构造测试数据
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 21128, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\n模型测试成功!")
    print(f"  label_4_logits shape: {outputs['label_4_logits'].shape}")
    print(f"  label_3_logits shape: {outputs['label_3_logits'].shape}")
    print(f"  arousal shape: {outputs['arousal'].shape}")
    print(f"  valence shape: {outputs['valence'].shape}")
    
    # 测试损失函数
    criterion = HMTLLoss()
    targets = {
        'label_4': torch.randint(0, 4, (batch_size,)),
        'label_3': torch.randint(0, 3, (batch_size,)),
        'arousal': torch.rand(batch_size),
        'valence': torch.rand(batch_size) * 2 - 1  # -1 to 1
    }
    
    loss_dict = criterion(outputs, targets)
    print(f"\n损失计算成功!")
    print(f"  total_loss: {loss_dict['total_loss']:.4f}")
    print(f"  loss_4: {loss_dict['loss_4']:.4f}")
    print(f"  loss_3: {loss_dict['loss_3']:.4f}")
    print(f"  loss_arousal: {loss_dict['loss_arousal']:.4f}")
    print(f"  loss_valence: {loss_dict['loss_valence']:.4f}")


if __name__ == "__main__":
    test_model()
