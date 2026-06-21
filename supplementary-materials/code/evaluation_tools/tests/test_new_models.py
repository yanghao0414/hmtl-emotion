#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新训练的视觉和音频模型
"""

import sys
import os
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

PROJECT_ROOT = bootstrap()

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import random

# 新训练的模型路径
VISUAL_MODEL_PATH = str(PROJECT_ROOT / "06_模型文件" / "visual_hmtl_trained.pt")
AUDIO_MODEL_PATH = str(PROJECT_ROOT / "06_模型文件" / "audio_hmtl_trained.pt")
DATA_ROOT = str(PROJECT_ROOT / "05_数据文件" / "visual_data_temp" / "archive (3)" / "Test")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 标签映射
LABEL_4_NAMES = ['积极', '激活消极', '非激活消极', '平静']
LABEL_7_NAMES = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']

# 数据集标签到4类映射
LABEL_MAPPING = {
    'happy': 0,      # 积极
    'Anger': 1,      # 激活消极
    'anger': 1,
    'fear': 1,
    'disgust': 1,
    'sad': 2,        # 非激活消极
    'neutral': 3,    # 平静
    'surprise': 0,
    'Contempt': 1,
    'contempt': 1,
}


class VisualEmotionModel(nn.Module):
    """视觉情绪识别模型"""
    
    def __init__(self, num_classes_4=4, num_classes_7=7, num_classes_3=3):
        super().__init__()
        
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.shared = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.head_4 = nn.Linear(128, num_classes_4)
        self.head_7 = nn.Linear(128, num_classes_7)
        self.head_3 = nn.Linear(128, num_classes_3)
        self.head_arousal = nn.Linear(128, 1)
        self.head_valence = nn.Linear(128, 1)
        
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        
        out_4 = self.head_4(shared)
        out_7 = self.head_7(shared)
        out_3 = self.head_3(shared)
        arousal = torch.tanh(self.head_arousal(shared))
        valence = torch.tanh(self.head_valence(shared))
        
        return out_4, out_7, out_3, arousal, valence


def load_visual_model():
    """加载新训练的视觉模型"""
    print("🔄 加载视觉模型...")
    
    model = VisualEmotionModel()
    checkpoint = torch.load(VISUAL_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✅ 视觉模型加载成功 (准确率: {checkpoint['accuracy_4']:.2%})")
    return model


def get_test_images(category, num_samples=10):
    """获取指定类别的测试图像"""
    category_path = os.path.join(DATA_ROOT, category)
    
    if not os.path.exists(category_path):
        return []
    
    images = []
    for f in os.listdir(category_path):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            images.append(os.path.join(category_path, f))
    
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    
    return images


def predict_image(model, image_path, transform):
    """预测单张图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out_4, out_7, out_3, arousal, valence = model(image_tensor)
            
            probs_4 = torch.softmax(out_4, dim=1)
            pred_4 = out_4.argmax(dim=1).item()
            confidence = probs_4[0, pred_4].item()
            
            pred_7 = out_7.argmax(dim=1).item()
        
        return {
            'pred_4': pred_4,
            'pred_7': pred_7,
            'confidence': confidence,
            'label_4': LABEL_4_NAMES[pred_4],
            'label_7': LABEL_7_NAMES[pred_7]
        }
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None


def run_test():
    """运行测试"""
    print("="*60)
    print("🎯 测试新训练的视觉模型")
    print("="*60)
    
    # 加载模型
    model = load_visual_model()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试各类别
    categories = ['happy', 'sad', 'Anger', 'neutral']
    all_results = []
    
    for category in categories:
        expected_4 = LABEL_MAPPING.get(category, 3)
        expected_name = LABEL_4_NAMES[expected_4]
        
        print(f"\n【{category}】 → 期望: {expected_name}")
        print("-"*40)
        
        images = get_test_images(category, num_samples=10)
        
        if not images:
            print(f"  ⚠️ 没有找到图像")
            continue
        
        correct = 0
        for img_path in images:
            result = predict_image(model, img_path, transform)
            
            if result:
                is_correct = result['pred_4'] == expected_4
                status = "✅" if is_correct else "❌"
                print(f"  {status} {os.path.basename(img_path)[:20]}... → {result['label_4']} ({result['confidence']:.2f})")
                
                all_results.append({
                    'category': category,
                    'expected': expected_4,
                    'predicted': result['pred_4'],
                    'correct': is_correct
                })
                
                if is_correct:
                    correct += 1
        
        print(f"  准确率: {correct}/{len(images)} = {correct/len(images):.1%}")
    
    # 总结
    if all_results:
        total_correct = sum(1 for r in all_results if r['correct'])
        total = len(all_results)
        accuracy = total_correct / total
        
        print("\n" + "="*60)
        print(f"📊 总体结果: {total_correct}/{total} = {accuracy:.1%}")
        print("="*60)
        
        # 按类别统计
        print("\n📈 各类别准确率:")
        for category in categories:
            cat_results = [r for r in all_results if r['category'] == category]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r['correct'])
                print(f"  {category}: {cat_correct}/{len(cat_results)} = {cat_correct/len(cat_results):.1%}")


if __name__ == "__main__":
    run_test()
