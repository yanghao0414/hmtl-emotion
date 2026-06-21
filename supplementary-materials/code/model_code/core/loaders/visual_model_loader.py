#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉模型兼容加载器
解决视觉模型架构不匹配的问题
"""

import torch
import torch.nn as nn
import torchvision.models as models
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class VisualHMTLModel(nn.Module):
    """
    视觉HMTL模型架构
    基于CNN backbone的多任务学习模型
    """
    
    def __init__(self, backbone_name='efficientnet_b0', num_classes_4=4, num_classes_3=3, num_classes_7=7):
        super().__init__()
        
        # 使用EfficientNet作为backbone
        if backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=False)
            # 移除最后的分类层
            self.backbone.classifier = nn.Identity()
            backbone_out_features = 1408  # 根据实际模型调整
        else:
            # 可以扩展支持其他backbone
            raise ValueError(f"不支持的backbone: {backbone_name}")
        
        # 共享特征层 - 根据实际模型调整
        self.shared_fc = nn.Sequential(
            nn.Linear(backbone_out_features, 512),  # 使用backbone_out_features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 4分类头 - 根据实际架构调整
        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes_4)
        )
        
        # 3分类头
        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes_3)
        )
        
        # 7分类头 - 根据实际架构调整
        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),  # 实际是256不是128
            nn.ReLU(),
            nn.Linear(256, num_classes_7)
        )
        
        # Arousal回归头 - 根据实际架构调整
        self.regressor_A = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),  # 实际是64不是128
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Valence回归头 - 根据实际架构调整
        self.regressor_V = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),  # 实际是64不是128
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, 3, H, W] 图像张量
            
        Returns:
            dict: 包含所有任务的输出
        """
        # 特征提取
        features = self.backbone(x)  # [batch_size, backbone_out_features]
        
        # 共享特征
        shared_features = self.shared_fc(features)  # [batch_size, 512]
        
        # 多任务输出
        label_4_logits = self.classifier_4(shared_features)
        label_3_logits = self.classifier_3(shared_features)
        label_7_logits = self.classifier_7(shared_features)
        arousal = self.regressor_A(shared_features).squeeze(-1)
        valence = self.regressor_V(shared_features).squeeze(-1)
        
        return {
            'label_4_logits': label_4_logits,
            'label_3_logits': label_3_logits,
            'label_7_logits': label_7_logits,
            'arousal': arousal,
            'valence': valence
        }


class VisualModelLoader:
    """视觉模型兼容加载器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.transform = None
        
    def load_model(self):
        """加载视觉模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"🔄 加载视觉模型: {os.path.basename(self.model_path)}")
        
        try:
            # 加载检查点
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 创建模型实例
            self.model = VisualHMTLModel()
            
            # 加载模型参数
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            
            # 定义图像预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ 视觉模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 视觉模型加载失败: {e}")
            return False
    
    def predict_from_image(self, image_path_or_tensor):
        """
        从图像文件或张量预测情绪
        
        Args:
            image_path_or_tensor: 图像文件路径或图像张量
            
        Returns:
            dict: 预测结果
        """
        if self.model is None or self.transform is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 处理图像输入
        if isinstance(image_path_or_tensor, str):
            # 从文件加载图像
            image = Image.open(image_path_or_tensor).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
        elif isinstance(image_path_or_tensor, torch.Tensor):
            image_tensor = image_path_or_tensor
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            # PIL Image
            image_tensor = self.transform(image_path_or_tensor).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 解析结果
        label_7_probs = torch.softmax(outputs['label_7_logits'], dim=1)[0]
        label_4_probs = torch.softmax(outputs['label_4_logits'], dim=1)[0]
        label_3_probs = torch.softmax(outputs['label_3_logits'], dim=1)[0]
        
        # 情绪标签映射
        emotion_7_names = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        emotion_4_names = ['积极', '激活消极', '非激活消极', '平静']
        emotion_3_names = ['积极', '消极', '平静']
        
        result = {
            '7类情绪': emotion_7_names[torch.argmax(label_7_probs).item()],
            '4类分类': emotion_4_names[torch.argmax(label_4_probs).item()],
            '3类极性': emotion_3_names[torch.argmax(label_3_probs).item()],
            'arousal': outputs['arousal'][0].item(),
            'valence': outputs['valence'][0].item()
        }
        
        return result
    
    def predict_from_text_placeholder(self, text):
        """
        临时方法：使用文本作为占位符生成随机预测
        实际应用中应该有图像输入
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 创建随机图像张量作为占位符
        import torch
        random_image = torch.randn(1, 3, 224, 224)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(random_image)
        
        # 解析结果（基于文本内容做一些简单的规则调整）
        label_7_probs = torch.softmax(outputs['label_7_logits'], dim=1)[0]
        label_4_probs = torch.softmax(outputs['label_4_logits'], dim=1)[0]
        label_3_probs = torch.softmax(outputs['label_3_logits'], dim=1)[0]
        
        # 根据文本内容调整预测（简单规则）
        if any(word in text for word in ['开心', '高兴', '好']):
            # 倾向于积极情绪
            label_4_idx = 0  # 积极
            label_7_idx = 2  # 快乐
        elif any(word in text for word in ['担心', '焦虑', '紧张']):
            # 倾向于激活消极
            label_4_idx = 1  # 激活消极
            label_7_idx = 1  # 焦虑
        elif any(word in text for word in ['难过', '悲伤', '失望']):
            # 倾向于非激活消极
            label_4_idx = 2  # 非激活消极
            label_7_idx = 3  # 悲伤
        else:
            # 使用模型原始预测
            label_4_idx = torch.argmax(label_4_probs).item()
            label_7_idx = torch.argmax(label_7_probs).item()
        
        emotion_7_names = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        emotion_4_names = ['积极', '激活消极', '非激活消极', '平静']
        emotion_3_names = ['积极', '消极', '平静']
        
        result = {
            '7类情绪': emotion_7_names[label_7_idx],
            '4类分类': emotion_4_names[label_4_idx],
            '3类极性': emotion_3_names[torch.argmax(label_3_probs).item()],
            'arousal': outputs['arousal'][0].item(),
            'valence': outputs['valence'][0].item()
        }
        
        return result


def test_visual_model():
    """测试视觉模型加载"""
    project_root = Path(__file__).resolve().parents[3]
    model_path = str(project_root / "06_模型文件" / "visual_hmtl_v4_best.pt")
    
    loader = VisualModelLoader(model_path)
    
    if loader.load_model():
        print("\n🎉 视觉模型测试成功！")
        print("模型已准备好处理图像输入")
        
        # 测试文本占位符预测
        try:
            result = loader.predict_from_text_placeholder("我今天心情很好")
            print("\n📊 测试预测结果:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"⚠️ 预测测试失败: {e}")
    else:
        print("❌ 视觉模型测试失败")


if __name__ == "__main__":
    test_visual_model()
