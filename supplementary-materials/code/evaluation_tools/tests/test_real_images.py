#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实面部表情图像测试多模态系统
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

import random
from PIL import Image
from multimodal_fusion_v2 import MultimodalFusionSystemV2


# 数据集路径
DATA_ROOT = str(PROJECT_ROOT / "05_数据文件" / "visual_data_temp" / "archive (3)" / "Test")

# 数据集标签到你的4类映射
LABEL_MAPPING = {
    'happy': '积极',
    'Anger': '激活消极',
    'anger': '激活消极',
    'fear': '激活消极',
    'disgust': '激活消极',
    'sad': '非激活消极',
    'neutral': '平静',
    'surprise': '积极',  # 惊讶可能是积极的
    'Contempt': '激活消极',
    'contempt': '激活消极',
}

# 数据集标签到7类映射
LABEL_7_MAPPING = {
    'happy': '快乐',
    'Anger': '愤怒',
    'anger': '愤怒',
    'fear': '焦虑',
    'disgust': '愤怒',
    'sad': '悲伤',
    'neutral': '平静',
    'surprise': '快乐',
    'Contempt': '愤怒',
    'contempt': '愤怒',
}


def get_test_images(category, num_samples=5):
    """获取指定类别的测试图像"""
    category_path = os.path.join(DATA_ROOT, category)
    
    if not os.path.exists(category_path):
        print(f"❌ 目录不存在: {category_path}")
        return []
    
    images = []
    for f in os.listdir(category_path):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            images.append(os.path.join(category_path, f))
    
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    
    return images


def test_single_image(system, image_path, expected_4, expected_7):
    """测试单张图像"""
    try:
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # 预测
        result = system.predict_visual(img)
        
        if result:
            pred_4 = result['4类分类']
            pred_7 = result['7类情绪']
            confidence = result['confidence']
            
            # 标准化标签比较
            is_correct_4 = pred_4 == expected_4 or \
                          (pred_4 == '激活型消极' and expected_4 == '激活消极') or \
                          (pred_4 == '非激活型消极' and expected_4 == '非激活消极')
            
            return {
                'image': os.path.basename(image_path),
                'expected_4': expected_4,
                'predicted_4': pred_4,
                'correct_4': is_correct_4,
                'expected_7': expected_7,
                'predicted_7': pred_7,
                'confidence': confidence
            }
    except Exception as e:
        print(f"  ❌ 错误: {e}")
    
    return None


def run_visual_evaluation(system):
    """运行视觉模型评估"""
    print("\n" + "="*60)
    print("📷 视觉模型真实图像评估")
    print("="*60)
    
    categories = ['happy', 'sad', 'Anger', 'neutral']
    all_results = []
    
    for category in categories:
        expected_4 = LABEL_MAPPING.get(category, '平静')
        expected_7 = LABEL_7_MAPPING.get(category, '平静')
        
        print(f"\n【{category}】 → 期望: {expected_4}/{expected_7}")
        print("-"*40)
        
        images = get_test_images(category, num_samples=5)
        
        if not images:
            print(f"  ⚠️ 没有找到图像")
            continue
        
        correct = 0
        for img_path in images:
            result = test_single_image(system, img_path, expected_4, expected_7)
            
            if result:
                all_results.append(result)
                status = "✅" if result['correct_4'] else "❌"
                print(f"  {status} {result['image'][:20]}... → {result['predicted_4']} (置信度: {result['confidence']:.2f})")
                
                if result['correct_4']:
                    correct += 1
        
        print(f"  准确率: {correct}/{len(images)}")
    
    # 总结
    if all_results:
        total_correct = sum(1 for r in all_results if r['correct_4'])
        total = len(all_results)
        accuracy = total_correct / total
        
        print("\n" + "="*60)
        print(f"📊 总体结果: {total_correct}/{total} = {accuracy:.1%}")
        print("="*60)
        
        return accuracy
    
    return 0


def run_multimodal_test(system):
    """运行多模态融合测试"""
    print("\n" + "="*60)
    print("🔗 多模态融合测试 (文本 + 图像)")
    print("="*60)
    
    test_cases = [
        {
            'text': '我今天心情很好',
            'image_category': 'happy',
            'expected_4': '积极'
        },
        {
            'text': '我很生气',
            'image_category': 'Anger',
            'expected_4': '激活消极'
        },
        {
            'text': '感觉很难过',
            'image_category': 'sad',
            'expected_4': '非激活消极'
        },
        {
            'text': '心情平静',
            'image_category': 'neutral',
            'expected_4': '平静'
        },
    ]
    
    for case in test_cases:
        print(f"\n【测试】文本: '{case['text']}' + 图像: {case['image_category']}")
        print("-"*40)
        
        # 获取一张图像
        images = get_test_images(case['image_category'], num_samples=1)
        if not images:
            print("  ⚠️ 没有图像")
            continue
        
        img = Image.open(images[0]).convert('RGB')
        
        # 多模态融合
        result = system.fuse(text=case['text'], image=img, strategy='weighted')
        
        if result and 'fusion_result' in result:
            fusion = result['fusion_result']
            
            # 显示各模态结果
            print("  单模态预测:")
            for mod, pred in result['single_predictions'].items():
                print(f"    {mod}: {pred['4类分类']} (置信度: {pred['confidence']:.2f})")
            
            print(f"  融合结果: {fusion['4类分类']} / {fusion['7类情绪']}")
            
            # 检查是否正确
            pred_4 = fusion['4类分类']
            is_correct = pred_4 == case['expected_4'] or \
                        (pred_4 == '激活型消极' and case['expected_4'] == '激活消极') or \
                        (pred_4 == '非激活型消极' and case['expected_4'] == '非激活消极')
            
            status = "✅ 正确" if is_correct else "❌ 错误"
            print(f"  {status} (期望: {case['expected_4']})")


def main():
    print("🚀 真实图像测试")
    print("="*60)
    
    # 初始化系统
    print("🔄 加载模型...")
    system = MultimodalFusionSystemV2()
    system.initialize_models(load_visual=True, load_audio=False)
    
    # 运行视觉评估
    run_visual_evaluation(system)
    
    # 运行多模态测试
    run_multimodal_test(system)
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
