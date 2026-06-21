#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实视觉模型加载器
使用预训练的图像模型进行情绪识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# 尝试导入torchvision
try:
    from torchvision import transforms, models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("⚠️ torchvision未安装，视觉模型功能受限")


class VisualEmotionModel(nn.Module):
    """
    视觉情绪识别模型 - 与训练脚本匹配的结构
    """
    
    def __init__(self, num_classes_4=4, num_classes_7=7, num_classes_3=3):
        super().__init__()
        
        # 使用预训练的ResNet18
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 多任务头
        self.head_4 = nn.Linear(128, num_classes_4)
        self.head_7 = nn.Linear(128, num_classes_7)
        self.head_3 = nn.Linear(128, num_classes_3)
        self.head_arousal = nn.Linear(128, 1)
        self.head_valence = nn.Linear(128, 1)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, 3, 224, 224] 图像张量
        Returns:
            tuple: (out_4, out_7, out_3, arousal, valence)
        """
        features = self.backbone(x)
        shared = self.shared(features)
        
        out_4 = self.head_4(shared)
        out_7 = self.head_7(shared)
        out_3 = self.head_3(shared)
        arousal = torch.tanh(self.head_arousal(shared))
        valence = torch.tanh(self.head_valence(shared))
        
        return out_4, out_7, out_3, arousal, valence


class RealVisualPredictor:
    """真实的视觉模型预测器"""
    
    def __init__(self, model_path=None, backbone='resnet18', device=None):
        self.model_path = model_path
        self.model = None
        self.loaded = False
        self.backbone = backbone
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_meta = {}
        
        # 图像预处理
        if TORCHVISION_AVAILABLE:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # 情绪标签映射
        self.emotion_4_names = ['积极', '激活消极', '非激活消极', '平静']
        self.emotion_7_names = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        self.emotion_3_names = ['积极', '消极', '平静']
    
    def load_model(self, use_pretrained=True):
        """
        加载模型
        """
        if not TORCHVISION_AVAILABLE:
            print("❌ torchvision未安装，无法加载视觉模型")
            return False
        
        try:
            print(f"🔄 初始化视觉模型...")

            # 加载训练好的权重
            if self.model_path and os.path.exists(self.model_path):
                print(f"🔄 加载训练好的权重: {self.model_path}")

                # 优先使用统一模型注册表，避免 V4 EfficientNet 权重被 ResNet 结构误加载。
                model_code_dir = Path(__file__).resolve().parents[2]
                if str(model_code_dir) not in sys.path:
                    sys.path.insert(0, str(model_code_dir))
                try:
                    from model_registry import load_model as load_registered_model

                    self.model, self.model_meta = load_registered_model(
                        self.model_path,
                        device=self.device,
                    )
                    arch = self.model_meta.get('architecture', 'unknown')
                    print(f"✅ 视觉模型加载成功 ({arch})")
                except Exception as registry_error:
                    print(f"⚠️ 模型注册表加载失败，回退到ResNet18兼容加载: {registry_error}")
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    self.model = VisualEmotionModel().to(self.device)

                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        acc_4 = checkpoint.get('accuracy_4', 0)
                        acc_7 = checkpoint.get('accuracy_7', 0)
                        print(f"✅ 模型加载成功 (4类准确率: {acc_4:.2%}, 7类准确率: {acc_7:.2%})")
                    else:
                        self.model.load_state_dict(checkpoint)
                        print("✅ 模型加载成功")
            else:
                print("⚠️ 未找到训练好的模型，使用随机初始化")
                self.model = VisualEmotionModel().to(self.device)
            
            self.model.eval()
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ 视觉模型加载失败: {e}")
            return False
    
    def predict_from_image(self, image_input):
        """
        从图像预测情绪
        Args:
            image_input: 图像路径(str)、PIL Image、或numpy数组
        Returns:
            dict: 预测结果
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 处理不同类型的输入
        if isinstance(image_input, str):
            # 从文件路径加载
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图像文件不存在: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # 从numpy数组
            image = Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # 已经是PIL Image
            image = image_input.convert('RGB')
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_input)}")
        
        # 预处理
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        
        # 预测：兼容旧tuple输出与注册表模型dict输出
        with torch.no_grad():
            outputs = self.model(input_tensor)

        if isinstance(outputs, dict):
            out_4 = outputs['label_4_logits']
            out_7 = outputs['label_7_logits']
            out_3 = outputs['label_3_logits']
            arousal = outputs['arousal']
            valence = outputs['valence']
        else:
            out_4, out_7, out_3, arousal, valence = outputs
        
        # 解析结果
        probs_4 = F.softmax(out_4[0], dim=0)
        probs_7 = F.softmax(out_7[0], dim=0)
        probs_3 = F.softmax(out_3[0], dim=0)
        
        pred_4 = torch.argmax(probs_4).item()
        pred_7 = torch.argmax(probs_7).item()
        pred_3 = torch.argmax(probs_3).item()
        
        # 计算置信度
        confidence = probs_4[pred_4].item()
        
        result = {
            '4类分类': self.emotion_4_names[pred_4],
            '7类情绪': self.emotion_7_names[pred_7],
            '3类极性': self.emotion_3_names[pred_3],
            'arousal': arousal.flatten()[0].item(),
            'valence': valence.flatten()[0].item(),
            'confidence': confidence,
            'probs_4': {self.emotion_4_names[i]: probs_4[i].item() for i in range(4)},
            'probs_3': {self.emotion_3_names[i]: probs_3[i].item() for i in range(3)},
            'probs_7': {self.emotion_7_names[i]: probs_7[i].item() for i in range(7)}
        }
        
        return result
    
    def predict_from_face(self, face_image):
        """
        从人脸图像预测情绪 (与predict_from_image相同，但语义更清晰)
        """
        return self.predict_from_image(face_image)
    
    def get_features(self, image_input):
        """
        提取图像特征 (用于融合)
        """
        if not self.loaded:
            raise RuntimeError("模型未加载")
        
        # 处理输入
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        if isinstance(outputs, dict):
            # 通用回退：返回多任务logits拼接，供诊断/融合特征实验使用。
            features = torch.cat([
                outputs['label_4_logits'].flatten(),
                outputs['label_7_logits'].flatten(),
                outputs['label_3_logits'].flatten(),
            ])
            return features.detach().cpu().numpy()

        out_4, out_7, out_3, _, _ = outputs
        features = torch.cat([out_4.flatten(), out_7.flatten(), out_3.flatten()])
        return features.detach().cpu().numpy()


def test_visual_model():
    """测试视觉模型"""
    print("🧪 测试真实视觉模型")
    print("="*50)
    
    if not TORCHVISION_AVAILABLE:
        print("❌ torchvision未安装，无法测试")
        return False
    
    # 创建预测器
    predictor = RealVisualPredictor(
        model_path=str(Path(__file__).resolve().parents[3] / "06_模型文件" / "visual_hmtl_v4_best.pt"),
        backbone='resnet18'
    )
    
    if not predictor.load_model():
        print("❌ 模型加载失败")
        return False
    
    # 创建测试图像 (随机噪声，仅用于测试模型是否工作)
    print("\n📷 创建测试图像...")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    try:
        result = predictor.predict_from_image(test_image)
        print("\n📊 预测结果:")
        print(f"  4类分类: {result['4类分类']}")
        print(f"  7类情绪: {result['7类情绪']}")
        print(f"  3类极性: {result['3类极性']}")
        print(f"  Arousal: {result['arousal']:.3f}")
        print(f"  Valence: {result['valence']:.3f}")
        print(f"  置信度: {result['confidence']:.3f}")
        
        print("\n✅ 视觉模型测试成功！")
        print("⚠️ 注意: 当前使用随机图像测试，需要真实面部图像才能获得有意义的结果")
        return True
        
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        return False


if __name__ == "__main__":
    test_visual_model()
