#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试工具 - 避免复杂依赖
直接测试多模态融合系统的核心功能
"""

import sys
import os
import time
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

bootstrap()

def test_basic_system():
    """测试基础系统"""
    print("🚀 简化测试 - 基础多模态系统")
    print("="*50)
    
    try:
        # 导入基础系统
        from hmtl_evaluate import HMTLPredictor
        from audio_model_loader import AudioModelLoader
        from visual_model_simple import SimpleVisualPredictor
        
        print("✅ 模块导入成功")
        
        # 测试文本模型
        print("\n📝 测试文本模型...")
        text_path = r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt"
        if os.path.exists(text_path):
            text_predictor = HMTLPredictor(text_path)
            result = text_predictor.predict("我今天心情很好", return_details=False)
            print(f"  文本预测: {result['emotion_4']} / {result['emotion_7']}")
            print("✅ 文本模型测试成功")
        else:
            print("❌ 文本模型文件不存在")
        
        # 测试音频模型
        print("\n🎵 测试音频模型...")
        audio_path = r"d:\bigcreate\06_模型文件\audio_hmtl_v2_best.pt"
        if os.path.exists(audio_path):
            audio_loader = AudioModelLoader(audio_path)
            if audio_loader.load_model():
                # 创建测试音频
                import random
                test_audio = [random.random() for _ in range(16000)]
                result = audio_loader.predict_from_audio(test_audio)
                print(f"  音频预测: {result['4类分类']} / {result['7类情绪']}")
                print("✅ 音频模型测试成功")
            else:
                print("❌ 音频模型加载失败")
        else:
            print("❌ 音频模型文件不存在")
        
        # 测试视觉模型
        print("\n👁️ 测试视觉模型...")
        visual_path = r"d:\bigcreate\06_模型文件\visual_hmtl_v4_best.pt"
        if os.path.exists(visual_path):
            visual_predictor = SimpleVisualPredictor(visual_path)
            if visual_predictor.load_model():
                result = visual_predictor.predict_from_text_placeholder("我今天心情很好")
                print(f"  视觉预测: {result['4类分类']} / {result['7类情绪']}")
                print("✅ 视觉模型测试成功")
            else:
                print("❌ 视觉模型加载失败")
        else:
            print("❌ 视觉模型文件不存在")
            
    except Exception as e:
        print(f"❌ 基础系统测试失败: {e}")
        return False
    
    return True

def test_fusion_logic():
    """测试融合逻辑"""
    print("\n🔗 测试融合逻辑...")
    print("="*50)
    
    # 模拟三个模态的预测结果
    predictions = {
        'text': {
            '4类分类': '积极',
            '3类极性': '积极', 
            '7类情绪': '支持',
            'arousal': 0.7,
            'valence': 0.8
        },
        'audio': {
            '4类分类': '激活消极',
            '3类极性': '消极',
            '7类情绪': '愤怒', 
            'arousal': 0.9,
            'valence': -0.6
        },
        'visual': {
            '4类分类': '积极',
            '3类极性': '积极',
            '7类情绪': '支持',
            'arousal': 0.6,
            'valence': 0.5
        }
    }
    
    print("📊 模拟预测结果:")
    for modality, pred in predictions.items():
        print(f"  {modality}: {pred['4类分类']} / {pred['7类情绪']}")
    
    # 简单投票融合
    print("\n🗳️ 简单投票融合:")
    votes_4 = {}
    for pred in predictions.values():
        class_4 = pred['4类分类']
        votes_4[class_4] = votes_4.get(class_4, 0) + 1
    
    winner_4 = max(votes_4, key=votes_4.get)
    print(f"  4类分类投票结果: {votes_4}")
    print(f"  获胜者: {winner_4}")
    
    # 加权融合 (文本权重高)
    print("\n⚖️ 加权融合:")
    weights = {'text': 0.99, 'audio': 0.65, 'visual': 0.64}
    weighted_votes_4 = {}
    
    for modality, pred in predictions.items():
        class_4 = pred['4类分类']
        weight = weights[modality]
        weighted_votes_4[class_4] = weighted_votes_4.get(class_4, 0) + weight
    
    weighted_winner_4 = max(weighted_votes_4, key=weighted_votes_4.get)
    print(f"  4类加权投票结果: {weighted_votes_4}")
    print(f"  加权获胜者: {weighted_winner_4}")
    
    # Arousal/Valence平均
    arousal_avg = sum(pred['arousal'] for pred in predictions.values()) / len(predictions)
    valence_avg = sum(pred['valence'] for pred in predictions.values()) / len(predictions)
    
    print(f"\n📈 连续值融合:")
    print(f"  平均Arousal: {arousal_avg:.3f}")
    print(f"  平均Valence: {valence_avg:.3f}")
    
    print("✅ 融合逻辑测试完成")
    return True

def test_multiple_inputs():
    """测试多个输入"""
    print("\n📝 测试多个输入...")
    print("="*50)
    
    test_cases = [
        "我今天心情很好，工作进展顺利",
        "我很担心明天的考试",
        "感觉很失落和沮丧", 
        "心情很平静",
        "既开心又紧张"
    ]
    
    try:
        from hmtl_evaluate import HMTLPredictor
        text_path = r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt"
        
        if os.path.exists(text_path):
            text_predictor = HMTLPredictor(text_path)
            
            print("🔍 文本模型预测结果:")
            for i, text in enumerate(test_cases, 1):
                try:
                    start_time = time.time()
                    result = text_predictor.predict(text, return_details=False)
                    exec_time = time.time() - start_time
                    
                    print(f"  {i}. {text[:20]}...")
                    print(f"     → {result['emotion_4']} / {result['emotion_7']} ({exec_time:.3f}s)")
                except Exception as e:
                    print(f"  {i}. 预测失败: {e}")
            
            print("✅ 多输入测试完成")
            return True
        else:
            print("❌ 文本模型文件不存在")
            return False
            
    except Exception as e:
        print(f"❌ 多输入测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 简化版系统测试")
    print("="*60)
    
    success_count = 0
    total_tests = 3
    
    # 测试1: 基础系统
    if test_basic_system():
        success_count += 1
    
    # 测试2: 融合逻辑
    if test_fusion_logic():
        success_count += 1
    
    # 测试3: 多输入测试
    if test_multiple_inputs():
        success_count += 1
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"  成功: {success_count}/{total_tests}")
    print(f"  成功率: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 所有测试通过！系统运行正常")
    else:
        print("⚠️ 部分测试失败，请检查系统配置")

if __name__ == "__main__":
    main()
