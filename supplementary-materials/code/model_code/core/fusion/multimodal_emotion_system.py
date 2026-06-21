#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的HMTL多模态情绪预测系统
整合文本、音频、视觉三个模态
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

bootstrap()

import torch
from hmtl_evaluate import HMTLPredictor
from audio_model_loader import AudioModelLoader
from visual_model_simple import SimpleVisualPredictor

class MultimodalEmotionSystem:
    """多模态情绪预测系统"""
    
    def __init__(self):
        """初始化三个模态的预测器"""
        self.text_predictor = None
        self.audio_loader = None
        self.visual_predictor = None
        
        # 模型路径配置
        self.model_paths = {
            'text': r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt",
            'audio': r"d:\bigcreate\06_模型文件\audio_hmtl_v2_best.pt",
            'visual': r"d:\bigcreate\06_模型文件\visual_hmtl_v4_best.pt"
        }
        
        self.available_modalities = []
        self._load_models()
    
    def _load_models(self):
        """加载所有可用的模型"""
        print("🔄 初始化多模态情绪预测系统...")
        print("="*60)
        
        # 加载文本模型
        try:
            if os.path.exists(self.model_paths['text']):
                print("📝 加载文本模型...")
                self.text_predictor = HMTLPredictor(self.model_paths['text'])
                self.available_modalities.append('text')
                print("✅ 文本模型加载成功")
            else:
                print("⚠️ 文本模型文件不存在")
        except Exception as e:
            print(f"❌ 文本模型加载失败: {e}")
        
        # 加载音频模型
        try:
            if os.path.exists(self.model_paths['audio']):
                print("🎵 加载音频模型...")
                self.audio_loader = AudioModelLoader(self.model_paths['audio'])
                if self.audio_loader.load_model():
                    self.available_modalities.append('audio')
                    print("✅ 音频模型加载成功")
                else:
                    print("❌ 音频模型加载失败")
            else:
                print("⚠️ 音频模型文件不存在")
        except Exception as e:
            print(f"❌ 音频模型加载失败: {e}")
        
        # 加载视觉模型
        try:
            if os.path.exists(self.model_paths['visual']):
                print("👁️ 加载视觉模型...")
                self.visual_predictor = SimpleVisualPredictor(self.model_paths['visual'])
                if self.visual_predictor.load_model():
                    self.available_modalities.append('visual')
                    print("✅ 视觉模型加载成功")
                else:
                    print("❌ 视觉模型加载失败")
            else:
                print("⚠️ 视觉模型文件不存在")
        except Exception as e:
            print(f"❌ 视觉模型加载失败: {e}")
        
        print(f"\n🎯 可用模态: {', '.join(self.available_modalities)}")
        print("="*60)
    
    def predict_text(self, text):
        """文本情绪预测"""
        if 'text' not in self.available_modalities:
            return None
        
        result = self.text_predictor.predict(text, return_details=False)
        return {
            '4类分类': result['emotion_4'],
            '3类极性': result['polarity_3'],
            '7类情绪': result['emotion_7'],
            'arousal': result['arousal'],
            'valence': result['valence'],
            '模态': 'text'
        }
    
    def predict_audio(self, audio_path_or_array, sample_rate=16000):
        """音频情绪预测"""
        if 'audio' not in self.available_modalities:
            return None
        
        result = self.audio_loader.predict_from_audio(audio_path_or_array, sample_rate)
        result['模态'] = 'audio'
        return result
    
    def predict_visual(self, text):
        """视觉情绪预测（当前使用文本输入作为占位符）"""
        if 'visual' not in self.available_modalities:
            return None
        
        result = self.visual_predictor.predict_from_text_placeholder(text)
        result['模态'] = 'visual'
        return result
    
    def predict_multimodal(self, text=None, audio=None, visual_text=None):
        """
        多模态融合预测
        
        Args:
            text: 文本输入
            audio: 音频输入（文件路径或数组）
            visual_text: 视觉输入（当前用文本代替）
            
        Returns:
            dict: 融合后的预测结果
        """
        results = {}
        predictions = []
        
        # 收集各模态预测
        if text is not None and 'text' in self.available_modalities:
            text_result = self.predict_text(text)
            if text_result:
                results['text'] = text_result
                predictions.append(text_result)
        
        if audio is not None and 'audio' in self.available_modalities:
            audio_result = self.predict_audio(audio)
            if audio_result:
                results['audio'] = audio_result
                predictions.append(audio_result)
        
        if visual_text is not None and 'visual' in self.available_modalities:
            visual_result = self.predict_visual(visual_text)
            if visual_result:
                results['visual'] = visual_result
                predictions.append(visual_result)
        
        if not predictions:
            return None
        
        # 简单的投票融合策略
        fusion_result = self._simple_fusion(predictions)
        results['fusion'] = fusion_result
        
        return results
    
    def _simple_fusion(self, predictions):
        """简单的多模态融合策略"""
        if not predictions:
            return None
        
        # 4类分类投票
        class_4_votes = {}
        for pred in predictions:
            class_4 = pred['4类分类']
            class_4_votes[class_4] = class_4_votes.get(class_4, 0) + 1
        
        # 3类极性投票
        class_3_votes = {}
        for pred in predictions:
            class_3 = pred['3类极性']
            class_3_votes[class_3] = class_3_votes.get(class_3, 0) + 1
        
        # 7类情绪投票
        class_7_votes = {}
        for pred in predictions:
            class_7 = pred['7类情绪']
            class_7_votes[class_7] = class_7_votes.get(class_7, 0) + 1
        
        # Arousal和Valence取平均
        arousal_avg = sum(pred['arousal'] for pred in predictions) / len(predictions)
        valence_avg = sum(pred['valence'] for pred in predictions) / len(predictions)
        
        return {
            '4类分类': max(class_4_votes, key=class_4_votes.get),
            '3类极性': max(class_3_votes, key=class_3_votes.get),
            '7类情绪': max(class_7_votes, key=class_7_votes.get),
            'arousal': round(arousal_avg, 3),
            'valence': round(valence_avg, 3),
            '模态': f"fusion({len(predictions)})"
        }
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            'available_modalities': self.available_modalities,
            'text_model': 'text' in self.available_modalities,
            'audio_model': 'audio' in self.available_modalities,
            'visual_model': 'visual' in self.available_modalities,
            'total_modalities': len(self.available_modalities)
        }
        return status


def demo():
    """演示多模态系统"""
    print("🎯 HMTL多模态情绪预测系统演示")
    print("="*60)
    
    # 初始化系统
    system = MultimodalEmotionSystem()
    
    # 显示系统状态
    status = system.get_system_status()
    print(f"\n📊 系统状态:")
    print(f"  可用模态数: {status['total_modalities']}/3")
    print(f"  文本模型: {'✅' if status['text_model'] else '❌'}")
    print(f"  音频模型: {'✅' if status['audio_model'] else '❌'}")
    print(f"  视觉模型: {'✅' if status['visual_model'] else '❌'}")
    
    # 测试文本预测
    if status['text_model']:
        print(f"\n📝 文本模态测试:")
        text_result = system.predict_text("我今天心情很好，工作进展顺利")
        if text_result:
            print(f"  输入: 我今天心情很好，工作进展顺利")
            print(f"  4类分类: {text_result['4类分类']}")
            print(f"  7类情绪: {text_result['7类情绪']}")
    
    # 测试音频预测
    if status['audio_model']:
        print(f"\n🎵 音频模态测试:")
        try:
            import numpy as np
            # 创建测试音频（1秒随机噪声）
            test_audio = np.random.randn(16000)
            audio_result = system.predict_audio(test_audio)
            if audio_result:
                print(f"  输入: 测试音频数据")
                print(f"  4类分类: {audio_result['4类分类']}")
                print(f"  7类情绪: {audio_result['7类情绪']}")
        except Exception as e:
            print(f"  音频测试失败: {e}")

    # 测试多模态融合
    if len(status['available_modalities']) > 1:
        print(f"\n🔗 多模态融合测试:")
        try:
            import numpy as np
            test_audio = np.random.randn(16000).tolist()  # 转换为列表避免数组比较问题

            multimodal_result = system.predict_multimodal(
                text="我很担心明天的考试",
                audio=test_audio if status['audio_model'] else None,
                visual_text="我很担心明天的考试" if status['visual_model'] else None
            )

            if multimodal_result and 'fusion' in multimodal_result:
                fusion = multimodal_result['fusion']
                print(f"  融合结果:")
                print(f"    4类分类: {fusion['4类分类']}")
                print(f"    7类情绪: {fusion['7类情绪']}")
                print(f"    使用模态: {fusion['模态']}")
        except Exception as e:
            print(f"  融合测试失败: {e}")


if __name__ == "__main__":
    demo()
