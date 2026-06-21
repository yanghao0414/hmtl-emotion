#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频模型兼容加载器
解决音频模型存储格式不同的问题
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
from pathlib import Path

class AudioHMTLModel(nn.Module):
    """
    音频HMTL模型架构
    基于Wav2Vec2的多任务学习模型
    """
    
    def __init__(self, wav2vec2_model_name='facebook/wav2vec2-base', dropout=0.3):
        super().__init__()
        
        # Wav2Vec2编码器
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_model_name)
        hidden_size = self.wav2vec2.config.hidden_size  # 768
        
        # 维度缩减层
        self.dim_reducer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU()
        )
        
        # 7分类头
        self.classifier_7 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7)
        )
        
        # 4分类头
        self.classifier_4 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )
        
        # 3分类头
        self.classifier_3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
        # Arousal回归头
        self.regressor_A = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Valence回归头
        self.regressor_V = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, input_values, attention_mask=None):
        """
        前向传播
        
        Args:
            input_values: [batch_size, sequence_length] 音频波形
            attention_mask: [batch_size, sequence_length] 注意力掩码
            
        Returns:
            dict: 包含所有任务的输出
        """
        # Wav2Vec2特征提取
        wav2vec2_outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        # 取平均池化作为句子级表示
        hidden_states = wav2vec2_outputs.last_hidden_state  # [batch_size, seq_len, 768]
        if attention_mask is not None:
            # 使用注意力掩码进行加权平均
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)  # [batch_size, 768]
        
        # 维度缩减
        reduced_features = self.dim_reducer(pooled_output)  # [batch_size, 256]
        
        # 多任务输出
        label_7_logits = self.classifier_7(reduced_features)
        label_4_logits = self.classifier_4(reduced_features)
        label_3_logits = self.classifier_3(reduced_features)
        arousal = self.regressor_A(reduced_features).squeeze(-1)
        valence = self.regressor_V(reduced_features).squeeze(-1)
        
        return {
            'label_7_logits': label_7_logits,
            'label_4_logits': label_4_logits,
            'label_3_logits': label_3_logits,
            'arousal': arousal,
            'valence': valence
        }


class AudioModelLoader:
    """音频模型兼容加载器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.processor = None
        
    def load_model(self):
        """加载音频模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"🔄 加载音频模型: {os.path.basename(self.model_path)}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 创建模型实例
            self.model = AudioHMTLModel()
            
            # 音频模型直接保存了参数，不是用model_state_dict包装
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 标准格式
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # 直接保存的参数格式
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            # 加载处理器
            self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
            
            print("✅ 音频模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 音频模型加载失败: {e}")
            return False
    
    def predict_from_audio(self, audio_path_or_array, sample_rate=16000):
        """
        从音频文件或数组预测情绪
        
        Args:
            audio_path_or_array: 音频文件路径或音频数组
            sample_rate: 采样率
            
        Returns:
            dict: 预测结果
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 处理音频输入
        if isinstance(audio_path_or_array, str):
            # 从文件加载音频
            import librosa
            audio_array, sr = librosa.load(audio_path_or_array, sr=sample_rate)
        else:
            audio_array = audio_path_or_array
        
        # 预处理音频
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # 预测
        with torch.no_grad():
            outputs = self.model(
                input_values=inputs.input_values,
                attention_mask=inputs.attention_mask if 'attention_mask' in inputs else None
            )
        
        # 解析结果
        label_7_probs = torch.softmax(outputs['label_7_logits'], dim=1)[0]
        label_4_probs = torch.softmax(outputs['label_4_logits'], dim=1)[0]
        label_3_probs = torch.softmax(outputs['label_3_logits'], dim=1)[0]
        
        # 情绪标签映射
        emotion_7_names = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        emotion_4_names = ['积极', '激活消极', '非激活消极', '平静']
        emotion_3_names = ['积极', '消极', '平静']

        pred_7 = torch.argmax(label_7_probs).item()
        pred_4 = torch.argmax(label_4_probs).item()
        pred_3 = torch.argmax(label_3_probs).item()
        
        result = {
            '7类情绪': emotion_7_names[pred_7],
            '4类分类': emotion_4_names[pred_4],
            '3类极性': emotion_3_names[pred_3],
            'arousal': outputs['arousal'][0].item(),
            'valence': outputs['valence'][0].item(),
            'confidence': label_4_probs[pred_4].item(),
            'probs_4': {emotion_4_names[i]: label_4_probs[i].item() for i in range(4)},
            'probs_3': {emotion_3_names[i]: label_3_probs[i].item() for i in range(3)},
            'probs_7': {emotion_7_names[i]: label_7_probs[i].item() for i in range(7)}
        }
        
        return result


def test_audio_model():
    """测试音频模型加载"""
    project_root = Path(__file__).resolve().parents[3]
    model_path = str(project_root / "06_模型文件" / "audio_hmtl_v2_best.pt")
    
    loader = AudioModelLoader(model_path)
    
    if loader.load_model():
        print("\n🎉 音频模型测试成功！")
        print("模型已准备好处理音频输入")
        
        # 创建测试音频（随机噪声）
        import numpy as np
        test_audio = np.random.randn(16000)  # 1秒的随机音频
        
        try:
            result = loader.predict_from_audio(test_audio)
            print("\n📊 测试预测结果:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠️ 预测测试失败: {e}")
    else:
        print("❌ 音频模型测试失败")


if __name__ == "__main__":
    test_audio_model()
