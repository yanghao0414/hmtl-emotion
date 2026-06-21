#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合策略演示
展示不同融合策略的效果
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

from PIL import Image
import random
from multimodal_fusion_v2 import MultimodalFusionSystemV2

# 测试数据路径
DATA_ROOT = str(PROJECT_ROOT / "05_数据文件" / "visual_data_temp" / "archive (3)" / "Test")


def get_sample_image(category):
    """获取一张样本图像"""
    category_path = os.path.join(DATA_ROOT, category)
    if os.path.exists(category_path):
        images = [f for f in os.listdir(category_path) if f.endswith(('.jpg', '.png'))]
        if images:
            return os.path.join(category_path, random.choice(images))
    return None


def demo_fusion_strategies():
    """演示不同融合策略"""
    print("="*70)
    print("🔗 多模态融合策略演示")
    print("="*70)
    
    # 初始化系统
    print("\n🔄 加载模型...")
    system = MultimodalFusionSystemV2()
    system.initialize_models(load_visual=True, load_audio=False)
    
    # 测试用例
    test_cases = [
        {
            'name': '一致性测试 - 开心',
            'text': '我今天心情很好，工作顺利',
            'image_category': 'happy',
            'expected': '积极'
        },
        {
            'name': '一致性测试 - 悲伤',
            'text': '感觉很失落和沮丧',
            'image_category': 'sad',
            'expected': '非激活消极'
        },
        {
            'name': '冲突测试 - 文本开心/图像悲伤',
            'text': '我很开心',
            'image_category': 'sad',
            'expected': '看融合策略'
        },
        {
            'name': '冲突测试 - 文本悲伤/图像开心',
            'text': '我很难过',
            'image_category': 'happy',
            'expected': '看融合策略'
        },
    ]
    
    strategies = ['weighted', 'hierarchical']
    
    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"📋 {case['name']}")
        print(f"   文本: '{case['text']}'")
        print(f"   图像: {case['image_category']}")
        print(f"   期望: {case['expected']}")
        print("-"*70)
        
        # 获取图像
        image_path = get_sample_image(case['image_category'])
        if not image_path:
            print("  ⚠️ 未找到图像")
            continue
        
        image = Image.open(image_path).convert('RGB')
        
        # 获取单模态预测
        text_pred = system.predict_text(case['text'])
        visual_pred = system.predict_visual(image)
        
        print("\n📊 单模态预测:")
        if text_pred:
            print(f"   文本: {text_pred['4类分类']} (置信度: {text_pred['confidence']:.2f})")
        if visual_pred:
            print(f"   视觉: {visual_pred['4类分类']} (置信度: {visual_pred['confidence']:.2f})")
        
        # 测试不同融合策略
        print("\n🔗 融合策略结果:")
        for strategy in strategies:
            result = system.fuse(
                text=case['text'],
                image=image,
                strategy=strategy
            )
            
            if result and 'fusion_result' in result:
                fusion = result['fusion_result']
                print(f"   {strategy:15s}: {fusion['4类分类']:10s} | 7类: {fusion['7类情绪']}")
    
    # 展示融合策略说明
    print("\n" + "="*70)
    print("📖 融合策略说明")
    print("="*70)
    
    print("""
┌─────────────────┬────────────────────────────────────────────────┐
│ 策略            │ 说明                                           │
├─────────────────┼────────────────────────────────────────────────┤
│ weighted        │ 加权投票：根据模型准确率加权                   │
│                 │ - 文本权重: 0.65                               │
│                 │ - 音频权重: 0.75                               │
│                 │ - 视觉权重: 0.70                               │
├─────────────────┼────────────────────────────────────────────────┤
│ hierarchical    │ 层级融合：先融合相似模态，再整体融合           │
│                 │ - 第一层: 音频+视觉 (非语言)                   │
│                 │ - 第二层: 文本 + 第一层结果                    │
├─────────────────┼────────────────────────────────────────────────┤
│ confidence      │ 置信度加权：高置信度的预测权重更大             │
│                 │ - 动态计算每个模态的置信度                     │
│                 │ - 置信度低的模态影响小                         │
└─────────────────┴────────────────────────────────────────────────┘

📌 当前系统使用的模型权重 (基于训练后准确率):
   - 文本模型: 65% → 权重 0.65
   - 音频模型: 75% → 权重 0.75  
   - 视觉模型: 70% → 权重 0.70
""")


def demo_single_prediction():
    """单次预测演示"""
    print("\n" + "="*70)
    print("🎯 单次预测演示")
    print("="*70)
    
    system = MultimodalFusionSystemV2()
    system.initialize_models(load_visual=True, load_audio=False)
    
    # 仅文本
    print("\n【仅文本预测】")
    result = system.fuse(text="我今天心情很好")
    if result:
        print(f"   结果: {result['fusion_result']['4类分类']}")
        print(f"   策略: {result['fusion_result']['strategy']}")
    
    # 文本+图像
    print("\n【文本+图像融合预测】")
    image_path = get_sample_image('happy')
    if image_path:
        image = Image.open(image_path).convert('RGB')
        result = system.fuse(text="我今天心情很好", image=image, strategy='weighted')
        if result:
            print(f"   文本预测: {result['single_predictions']['text']['4类分类']}")
            print(f"   视觉预测: {result['single_predictions']['visual']['4类分类']}")
            print(f"   融合结果: {result['fusion_result']['4类分类']}")
            print(f"   融合策略: {result['fusion_result']['strategy']}")


if __name__ == "__main__":
    demo_fusion_strategies()
    demo_single_prediction()
    print("\n✅ 演示完成！")
