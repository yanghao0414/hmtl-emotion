"""
Visual HMTL 视觉情绪识别分类器
基于 EfficientNet-B2 的视觉情绪识别

输出格式与 Text/Audio HMTL 保持一致:
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
import torchvision.models as models


class VisualHMTLClassifier(nn.Module):
    """
    Visual HMTL 视觉情绪分类器
    基于 EfficientNet-B2 预训练模型
    """
    
    def __init__(self, dropout=0.3, pretrained=True):
        super().__init__()
        
        # EfficientNet-B2 骨干网络
        if pretrained:
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.efficientnet_b2(weights=None)
        
        # 获取特征维度 (EfficientNet-B2: 1408)
        feature_dim = self.backbone.classifier[1].in_features
        
        # 移除原始分类头
        self.backbone.classifier = nn.Identity()
        
        # 共享特征层
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 任务1: 7类情绪
        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 7)
        )
        
        # 任务2: 4类情绪
        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4)
        )
        
        # 任务3: 3类极性
        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 3)
        )
        
        # 任务4: Arousal 唤醒度回归
        self.regressor_A = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出 0-1
        )
        
        # 任务5: Valence 效价回归
        self.regressor_V = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # 输出 -1 到 1
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, 3, 224, 224]
        
        Returns:
            dict: 多任务预测结果
        """
        # 提取特征
        features = self.backbone(x)  # [batch_size, 1408]
        
        # 共享特征层
        shared = self.shared_fc(features)  # [batch_size, 512]
        
        # 多任务预测
        logits_7 = self.classifier_7(shared)
        logits_4 = self.classifier_4(shared)
        logits_3 = self.classifier_3(shared)
        arousal = self.regressor_A(shared).squeeze(-1)
        valence = self.regressor_V(shared).squeeze(-1)
        
        return {
            'label_7_logits': logits_7,
            'label_4_logits': logits_4,
            'label_3_logits': logits_3,
            'arousal': arousal,
            'valence': valence
        }


def test_model():
    """测试模型"""
    print("测试 Visual HMTL 分类器...")
    
    model = VisualHMTLClassifier(pretrained=False)
    model.eval()
    
    # 构造测试输入
    x = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(x)
    
    print("\n输出形状:")
    for key, value in outputs.items():
        print(f"  {key}: shape={value.shape}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params/1e6:.2f}M")
    print(f"可训练参数: {trainable_params/1e6:.2f}M")


if __name__ == "__main__":
    test_model()
