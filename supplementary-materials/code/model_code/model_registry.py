#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型注册表 — 统一管理所有模型版本的架构定义和加载逻辑

支持的模型版本:
  文本: HMTLEmotionModelV2 (best_model_v2.pt), HMTLEmotionModelV3 (text_hmtl_v3_best.pt)
  视觉: VisualHMTLClassifier V1/V2/V3/V4 + VisualEmotionModel(ResNet18)
  音频: AudioHMTLClassifier (wav2vec2), AudioMLP (手工特征)

使用方法:
    from model_registry import load_model
    model = load_model('text', 'path/to/model.pt')
    model = load_model('visual', 'path/to/model.pt')
    model = load_model('audio', 'path/to/model.pt')
"""

import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# 文本模型
# ============================================================

class HMTLEmotionModelV3(nn.Module):
    """
    V3版文本模型 — arousal/valence头是简化的2层结构
    对应: text_hmtl_v3_best.pt
    与V2的区别: arousal_head/valence_head 为 768→128→1 (V2是768→128→64→1)
    """

    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.3, num_emotions=7):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size  # 768

        # 任务1: 4类情绪分类 (与V2相同)
        self.classifier_4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, 4)
        )

        # 任务2: 3类情感极性 (与V2相同)
        self.classifier_3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, 3)
        )

        # 任务3: 7类细粒度情绪 (与V2相同)
        self.classifier_7 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),
            nn.Linear(256, num_emotions)
        )

        # 任务4: Arousal唤醒度 — V3简化版(2层)
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 任务5: Valence效价 — V3简化版(2层)
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return {
            'label_4_logits': self.classifier_4(cls_output),
            'label_3_logits': self.classifier_3(cls_output),
            'label_7_logits': self.classifier_7(cls_output),
            'arousal': self.arousal_head(cls_output).squeeze(-1),
            'valence': self.valence_head(cls_output).squeeze(-1)
        }


# ============================================================
# 视觉模型
# ============================================================

class VisualHMTLClassifierV2V3(nn.Module):
    """
    V2/V3版视觉模型 — 与V1相比 classifier_4/classifier_3 隐藏层更宽
    对应: visual_hmtl_v2_best.pt, visual_hmtl_v3_best.pt

    与V1差异:
      classifier_4: Linear(512,256) vs V1的Linear(512,128)
      classifier_3: Linear(512,128) vs V1的Linear(512,64)
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.efficientnet_b2(weights=None)
        feature_dim = self.backbone.classifier[1].in_features  # 1408
        self.backbone.classifier = nn.Identity()

        # 共享特征层 (与V1相同)
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # 任务1: 7类情绪 (与V1相同: 512→256→7)
        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 7)
        )

        # 任务2: 4类情绪 — V2/V3加宽: 512→256→4 (V1是512→128→4)
        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 4)
        )

        # 任务3: 3类极性 — V2/V3加宽: 512→128→3 (V1是512→64→3)
        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3)
        )

        # 任务4: Arousal唤醒度回归 (与V1相同)
        self.regressor_A = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 任务5: Valence效价回归 (与V1相同)
        self.regressor_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        return {
            'label_7_logits': self.classifier_7(shared),
            'label_4_logits': self.classifier_4(shared),
            'label_3_logits': self.classifier_3(shared),
            'arousal': self.regressor_A(shared).squeeze(-1),
            'valence': self.regressor_V(shared).squeeze(-1)
        }


class VisualHMTLClassifierV4(nn.Module):
    """
    V4版视觉模型 — 在V2/V3基础上加入BatchNorm
    对应: visual_hmtl_v4_best.pt
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.efficientnet_b2(weights=None)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 7)
        )

        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 4)
        )

        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3)
        )

        self.regressor_A = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.regressor_V = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        return {
            'label_7_logits': self.classifier_7(shared),
            'label_4_logits': self.classifier_4(shared),
            'label_3_logits': self.classifier_3(shared),
            'arousal': self.regressor_A(shared).squeeze(-1),
            'valence': self.regressor_V(shared).squeeze(-1)
        }


class VisualEmotionModel(nn.Module):
    """
    ResNet18版视觉模型
    对应: visual_hmtl_trained.pt
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        nf = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.shared = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.head_4 = nn.Linear(128, 4)
        self.head_7 = nn.Linear(128, 7)
        self.head_3 = nn.Linear(128, 3)
        self.head_arousal = nn.Linear(128, 1)
        self.head_valence = nn.Linear(128, 1)

    def forward(self, x):
        f = self.backbone(x)
        s = self.shared(f)
        return {
            'label_4_logits': self.head_4(s),
            'label_7_logits': self.head_7(s),
            'label_3_logits': self.head_3(s),
            'arousal': self.head_arousal(s).squeeze(-1),
            'valence': self.head_valence(s).squeeze(-1)
        }


# ============================================================
# 音频模型
# ============================================================

class AudioMLP(nn.Module):
    """
    基于手工特征的音频MLP分类器
    对应: audio_hmtl_trained.pt (0.26MB, 65K params, 输入82维)
    注意: 这不是Wav2Vec2模型，需要预先提取82维手工特征
    """

    def __init__(self, in_dim=82):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64))
        self.head_4 = nn.Linear(64, 4)
        self.head_3 = nn.Linear(64, 3)
        self.head_7 = nn.Linear(64, 7)
        self.head_arousal = nn.Linear(64, 1)
        self.head_valence = nn.Linear(64, 1)

    def forward(self, x):
        f = self.encoder(x)
        return {
            'label_4_logits': self.head_4(f),
            'label_7_logits': self.head_7(f),
            'label_3_logits': self.head_3(f),
            'arousal': self.head_arousal(f).squeeze(-1),
            'valence': self.head_valence(f).squeeze(-1)
        }


# ============================================================
# 自动检测 & 加载
# ============================================================

# 模型注册表: filename → (架构类, 说明, 推荐度)
MODEL_REGISTRY = {
    # 文本模型
    'best_model_v2.pt':          ('HMTLEmotionModelV2', '主力文本模型(SMP2020, 78.7%)', '★★★'),
    'smp2020_final_model.pt':    ('HMTLEmotionModelV2', 'SMP2020文本模型(78.7%)', '★★★'),
    'text_hmtl_v3_best.pt':     ('HMTLEmotionModelV3', 'V3文本模型(过拟合,真实60-66%)', '★☆☆'),
    # 视觉模型
    'visual_hmtl_v4_best.pt':   ('VisualHMTLClassifierV4', 'V4视觉EfficientNet+BN(62.9%)', '★★★'),
    'visual_hmtl_trained.pt':   ('VisualEmotionModel', 'ResNet18视觉(69.8%)', '★★☆'),
    'visual_hmtl_best.pt':      ('VisualHMTLClassifier', 'V1视觉EfficientNet', '★★☆'),
    'visual_hmtl_v2_best.pt':   ('VisualHMTLClassifierV2V3', 'V2视觉EfficientNet(37.2%)', '★☆☆'),
    'visual_hmtl_v3_best.pt':   ('VisualHMTLClassifierV2V3', 'V3视觉EfficientNet(58.5%)', '★★☆'),
    # 音频模型
    'audio_hmtl_v2_best.pt':    ('AudioHMTLClassifier', 'Wav2Vec2音频(75.4%)', '★★★'),
    'audio_best_hmtl.pt':       ('AudioHMTLClassifier', 'Wav2Vec2音频(早期版本)', '★★☆'),
    'audio_hmtl_trained.pt':    ('AudioMLP', '手工特征MLP音频(需82维输入)', '★☆☆'),
}

# 推荐的官方模型组合
OFFICIAL_MODELS = {
    'text':   'best_model_v2.pt',         # 或 smp2020_final_model.pt
    'visual': 'visual_hmtl_v4_best.pt',   # 论文使用的V4
    'audio':  'audio_hmtl_v2_best.pt',    # 完整Wav2Vec2
}


def detect_architecture(state_dict):
    """
    根据state_dict的key自动检测模型架构

    Returns:
        str: 架构类名
    """
    keys = set(state_dict.keys())

    # 文本模型判断
    if any('bert.' in k for k in keys):
        if 'arousal_head.4.weight' in keys:
            return 'HMTLEmotionModelV2'    # 3层arousal head
        elif 'arousal_head.3.weight' in keys:
            return 'HMTLEmotionModelV3'    # 2层arousal head
        else:
            return 'HMTLEmotionModelV2'    # 默认V2

    # 音频模型判断
    if any('wav2vec2' in k for k in keys):
        return 'AudioHMTLClassifier'
    if 'encoder.0.weight' in keys and not any('backbone' in k for k in keys):
        return 'AudioMLP'

    # 视觉模型判断
    has_backbone = any('backbone.' in k for k in keys)
    if has_backbone:
        has_effnet = any('backbone.features' in k for k in keys)
        has_resnet = any('backbone.layer1' in k for k in keys)

        if has_resnet:
            return 'VisualEmotionModel'

        if has_effnet:
            # 区分V1/V2V3/V4
            has_shared_bn = 'shared_fc.3.weight' in keys  # BatchNorm1d has .weight
            if has_shared_bn:
                return 'VisualHMTLClassifierV4'

            # V1 vs V2V3: check classifier_4 hidden dim
            if 'classifier_4.0.weight' in keys:
                c4_shape = state_dict['classifier_4.0.weight'].shape
                if c4_shape[0] == 256:
                    return 'VisualHMTLClassifierV2V3'
                elif c4_shape[0] == 128:
                    return 'VisualHMTLClassifier'

            return 'VisualHMTLClassifier'  # 默认V1

    return None


def _get_model_class(class_name):
    """获取模型类"""
    classes = {
        'HMTLEmotionModelV3': HMTLEmotionModelV3,
        'VisualHMTLClassifierV2V3': VisualHMTLClassifierV2V3,
        'VisualHMTLClassifierV4': VisualHMTLClassifierV4,
        'VisualEmotionModel': VisualEmotionModel,
        'AudioMLP': AudioMLP,
    }
    if class_name in classes:
        return classes[class_name]

    # 延迟导入已有模型
    if class_name == 'HMTLEmotionModelV2':
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from hmtl_model_v2 import HMTLEmotionModelV2
        return HMTLEmotionModelV2
    if class_name == 'VisualHMTLClassifier':
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'visual_hmtl'))
        from visual_hmtl_classifier import VisualHMTLClassifier
        return VisualHMTLClassifier
    if class_name == 'AudioHMTLClassifier':
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'audio_hmtl'))
        from audio_hmtl_classifier import AudioHMTLClassifier
        return AudioHMTLClassifier

    raise ValueError(f"未知模型类: {class_name}")


def load_model(model_path, device='cpu', arch_override=None):
    """
    自动检测架构并加载模型

    Args:
        model_path: 模型文件路径
        device: 设备
        arch_override: 强制指定架构类名 (可选)

    Returns:
        (model, meta): 加载好的模型和元信息dict
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        meta = {k: v for k, v in ckpt.items()
                if k not in ('model_state_dict', 'optimizer_state_dict')
                and not isinstance(v, dict)}
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError(f"无法解析的checkpoint格式: {type(ckpt)}")

    # 确定架构
    if arch_override:
        arch_name = arch_override
    else:
        filename = Path(model_path).name
        if filename in MODEL_REGISTRY:
            arch_name = MODEL_REGISTRY[filename][0]
        else:
            arch_name = detect_architecture(state_dict)

    if arch_name is None:
        raise ValueError(f"无法自动检测架构: {model_path}")

    # 创建模型并加载权重
    ModelClass = _get_model_class(arch_name)

    # 特殊处理: 带pretrained参数的类
    if arch_name in ('VisualHMTLClassifier',):
        model = ModelClass(pretrained=False)
    else:
        model = ModelClass()

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    meta['architecture'] = arch_name
    return model, meta


# ============================================================
# 测试
# ============================================================

if __name__ == '__main__':
    import os
    model_dir = os.path.join(os.path.dirname(__file__), '..', '06_模型文件')

    print("=" * 70)
    print("模型注册表 — 全量加载测试")
    print("=" * 70)

    test_files = [
        os.path.join(model_dir, 'hmtl_models_v2', 'best_model_v2.pt'),
        os.path.join(model_dir, 'text_hmtl_v3_best.pt'),
        os.path.join(model_dir, 'visual_hmtl_v4_best.pt'),
        os.path.join(model_dir, 'visual_hmtl_trained.pt'),
        os.path.join(model_dir, 'visual_hmtl_best.pt'),
        os.path.join(model_dir, 'visual_hmtl_v2_best.pt'),
        os.path.join(model_dir, 'visual_hmtl_v3_best.pt'),
        os.path.join(model_dir, 'audio_hmtl_v2_best.pt'),
        os.path.join(model_dir, 'audio_hmtl_trained.pt'),
    ]

    for fp in test_files:
        name = os.path.relpath(fp, model_dir)
        try:
            model, meta = load_model(fp)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ {name}: {meta.get('architecture','?')} ({n_params/1e6:.1f}M params) {meta}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
