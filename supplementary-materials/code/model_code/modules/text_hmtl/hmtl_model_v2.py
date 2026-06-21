#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL V2 - 改进版多任务情绪识别模型
改进点:
1. Focal Loss 解决类别不平衡
2. Arousal回归增强
3. 新增7类细粒度情绪分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class FocalLoss(nn.Module):
    """
    Focal Loss for 7-class emotion classification
    解决类别不平衡问题
    
    参考: Focal Loss for Dense Object Detection (Lin et al., 2017)
    公式: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重 [num_classes] 或 None
            gamma: 聚焦参数，默认2.0
            reduction: 归约方式 'mean', 'sum' 或 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # p_t = probability of correct class
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                alpha_t = self.alpha[targets]
                focal_weight = alpha_t * focal_weight
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HMTLEmotionModelV2(nn.Module):
    """
    HMTL V2 改进版多任务情绪识别模型
    
    改进点:
    - 新增7类细粒度情绪分类任务
    - 使用Focal Loss解决类别不平衡
    - 增强的回归头设计
    """
    
    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.3, num_emotions=7):
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
        
        # 任务3: 7类细粒度情绪
        self.classifier_7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, num_emotions)
        )
        
        # 任务4: Arousal唤醒度
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1
        )
        
        # 任务5: Valence效价
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出-1到1
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Returns:
            dict: {
                'label_4_logits': [batch_size, 4],
                'label_3_logits': [batch_size, 3],
                'label_7_logits': [batch_size, 7],  # 新增
                'arousal': [batch_size],
                'valence': [batch_size]
            }
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # 多任务预测
        label_4_logits = self.classifier_4(cls_output)
        label_3_logits = self.classifier_3(cls_output)
        label_7_logits = self.classifier_7(cls_output)  # 新增7类
        arousal = self.arousal_head(cls_output).squeeze(-1)
        valence = self.valence_head(cls_output).squeeze(-1)
        
        return {
            'label_4_logits': label_4_logits,
            'label_3_logits': label_3_logits,
            'label_7_logits': label_7_logits,
            'arousal': arousal,
            'valence': valence
        }


class HMTLLossV2(nn.Module):
    """
    HMTL V2 多任务损失函数
    
    改进点:
    - 7类使用Focal Loss
    - Arousal权重增强
    - 支持类别权重
    """
    
    def __init__(self, 
                 weight_4=1.0,           # 4类分类权重
                 weight_3=0.8,           # 3类分类权重
                 weight_7=1.2,           # 7类分类权重
                 weight_arousal=0.8,     # Arousal回归权重
                 weight_valence=0.5,     # Valence回归权重
                 use_focal_loss=True,    # 是否使用Focal Loss
                 focal_gamma=2.0,        # Focal Loss gamma参数
                 class_weights_4=None,
                 class_weights_3=None,
                 class_weights_7=None):
        super().__init__()
        self.weight_4 = weight_4
        self.weight_3 = weight_3
        self.weight_7 = weight_7
        self.weight_arousal = weight_arousal
        self.weight_valence = weight_valence
        
        # 分类损失
        self.ce_loss_4 = nn.CrossEntropyLoss(weight=class_weights_4)
        self.ce_loss_3 = nn.CrossEntropyLoss(weight=class_weights_3)
        
        # 7类使用Focal Loss
        if use_focal_loss:
            self.ce_loss_7 = FocalLoss(alpha=class_weights_7, gamma=focal_gamma)
            self.loss_7_name = 'FocalLoss'
        else:
            self.ce_loss_7 = nn.CrossEntropyLoss(weight=class_weights_7)
            self.loss_7_name = 'CrossEntropy'
        
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
                'label_7': [batch_size],  # 新增
                'arousal': [batch_size],
                'valence': [batch_size]
            }
        """
        # 分类损失
        loss_4 = self.ce_loss_4(outputs['label_4_logits'], targets['label_4'])
        loss_3 = self.ce_loss_3(outputs['label_3_logits'], targets['label_3'])
        loss_7 = self.ce_loss_7(outputs['label_7_logits'], targets['label_7'])
        
        # 回归损失
        loss_arousal = self.mse_loss(outputs['arousal'], targets['arousal'])
        loss_valence = self.mse_loss(outputs['valence'], targets['valence'])
        
        # 加权总损失
        total_loss = (self.weight_4 * loss_4 + 
                     self.weight_3 * loss_3 +
                     self.weight_7 * loss_7 +
                     self.weight_arousal * loss_arousal + 
                     self.weight_valence * loss_valence)
        
        return {
            'total_loss': total_loss,
            'loss_4': loss_4.item(),
            'loss_3': loss_3.item(),
            'loss_7': loss_7.item(),
            'loss_arousal': loss_arousal.item(),
            'loss_valence': loss_valence.item()
        }


def test_model_v2():
    """测试V2模型"""
    print("测试HMTL V2模型...")
    
    # 创建模型
    model = HMTLEmotionModelV2()
    model.eval()
    
    # 构造测试数据
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 21128, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"\n模型测试成功 V2!")
    print(f"  label_4_logits shape: {outputs['label_4_logits'].shape}")
    print(f"  label_3_logits shape: {outputs['label_3_logits'].shape}")
    print(f"  label_7_logits shape: {outputs['label_7_logits'].shape}")
    print(f"  arousal shape: {outputs['arousal'].shape}")
    print(f"  valence shape: {outputs['valence'].shape}")
    
    # 测试损失函数
    criterion = HMTLLossV2(
        use_focal_loss=True,
        focal_gamma=2.0,
        weight_arousal=0.8  # 增强Arousal权重
    )
    
    targets = {
        'label_4': torch.randint(0, 4, (batch_size,)),
        'label_3': torch.randint(0, 3, (batch_size,)),
        'label_7': torch.randint(0, 7, (batch_size,)),
        'arousal': torch.rand(batch_size),
        'valence': torch.rand(batch_size) * 2 - 1
    }
    
    loss_dict = criterion(outputs, targets)
    print(f"\n损失计算成功 V2!")
    print(f"  total_loss: {loss_dict['total_loss']:.4f}")
    print(f"  loss_7 (Focal): {loss_dict['loss_7']:.4f}")
    print(f"  loss_arousal: {loss_dict['loss_arousal']:.4f}")


if __name__ == "__main__":
    test_model_v2()
