"""
Audio HMTL 音频情绪分类器 V2
输出格式与 Text HMTL 保持一致

输出格式:
{
    'label_7_logits': [batch_size, 7],   # 7类情绪
    'label_4_logits': [batch_size, 4],   # 4类情绪
    'label_3_logits': [batch_size, 3],   # 3类极性
    'arousal': [batch_size],             # 唤醒度
    'valence': [batch_size]              # 效价
}
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import os, sys

# 导入 label_mapper 工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
from label_mapper import NUM_4_EMOTION_CLASSES, NUM_3_POLARITY_CLASSES

WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base"
NUM_7_EMOTION_CLASSES = 7  # 细粒度情绪类别数


class AudioHMTLClassifier(nn.Module):
    """
    Audio HMTL 音频情绪分类器 V2
    基于 Wav2Vec2 预训练模型
    """
    
    def __init__(self, dropout=0.3):
        super(AudioHMTLClassifier, self).__init__()
        
        # Wav2Vec2 
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
        self.wav2vec2.freeze_feature_extractor()

        config = Wav2Vec2Config.from_pretrained(WAV2VEC2_MODEL_NAME)
        hidden_size = config.hidden_size  # 768
        
        # 维度压缩层
        self.dim_reducer = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        reduced_size = 256

        # 任务1: 7类情绪 (细粒度)
        self.classifier_7 = nn.Sequential(
            nn.Linear(reduced_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, NUM_7_EMOTION_CLASSES)
        )
        
        # 任务2: 4类情绪
        self.classifier_4 = nn.Sequential(
            nn.Linear(reduced_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(64, NUM_4_EMOTION_CLASSES)
        )
        
        # 任务3: 3类极性
        self.classifier_3 = nn.Sequential(
            nn.Linear(reduced_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(64, NUM_3_POLARITY_CLASSES)
        )
        
        # 任务4: Arousal 唤醒度回归
        self.regressor_A = nn.Sequential(
            nn.Linear(reduced_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 0-1
        )
        
        # 任务5: Valence 效价回归
        self.regressor_V = nn.Sequential(
            nn.Linear(reduced_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # 输出 -1 到 1
        )

    def forward(self, input_values, attention_mask=None):
        """
        前向传播
        
        Args:
            input_values: 音频输入 [batch_size, sequence_length]
            attention_mask: 注意力掩码 [batch_size, sequence_length]
        
        Returns:
            dict: 多任务预测结果
        """
        # Wav2Vec2 编码
        output = self.wav2vec2(input_values, attention_mask=attention_mask)
        
        # Mean Pooling
        mean_pooling = torch.mean(output.last_hidden_state, dim=1)
        
        # 维度压缩
        x = self.dim_reducer(mean_pooling)

        # 多任务预测
        logits_7 = self.classifier_7(x)
        logits_4 = self.classifier_4(x)
        logits_3 = self.classifier_3(x)
        arousal = self.regressor_A(x).squeeze(-1)
        valence = self.regressor_V(x).squeeze(-1)

        return {
            'label_7_logits': logits_7,
            'label_4_logits': logits_4,
            'label_3_logits': logits_3,
            'arousal': arousal,
            'valence': valence
        }


# 旧版兼容
class AudioHMTLClassifierLegacy(nn.Module):
    """旧版音频分类器兼容接口"""
    
    def __init__(self):
        super(AudioHMTLClassifierLegacy, self).__init__()
        
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
        self.wav2vec2.freeze_feature_extractor()

        config = Wav2Vec2Config.from_pretrained(WAV2VEC2_MODEL_NAME)
        hidden_size = config.hidden_size
        
        self.dim_reducer = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        reduced_size = 256

        self.classifier_4 = nn.Linear(reduced_size, NUM_4_EMOTION_CLASSES)
        self.classifier_3 = nn.Linear(reduced_size, NUM_3_POLARITY_CLASSES)
        self.regressor_A = nn.Linear(reduced_size, 1)
        self.regressor_V = nn.Linear(reduced_size, 1)                      

    def forward(self, input_values, attention_mask):
        output = self.wav2vec2(input_values, attention_mask=attention_mask)
        mean_pooling = torch.mean(output.last_hidden_state, dim=1)
        x = self.relu(self.dim_reducer(mean_pooling))

        logits_4 = self.classifier_4(x)
        logits_3 = self.classifier_3(x)
        pred_A = self.regressor_A(x)
        pred_V = self.regressor_V(x)

        return logits_4, logits_3, pred_A, pred_V
