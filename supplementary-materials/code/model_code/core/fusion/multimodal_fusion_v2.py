#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合系统 V2 - 修复版
解决了以下问题:
1. 视觉模型使用真实的图像处理
2. 音频模型支持真实音频输入
3. 动态置信度计算
4. 真实的模型权重
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
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union
from PIL import Image

# 导入各模态模型
from hmtl_evaluate import HMTLPredictor
from audio_model_loader import AudioModelLoader
from visual_model_real import RealVisualPredictor, TORCHVISION_AVAILABLE


class MultimodalFusionSystemV2:
    """
    多模态融合系统 V2 - 修复版
    
    改进:
    1. 支持真实的多模态输入 (文本、音频文件、图像文件)
    2. 动态置信度计算
    3. 基于真实性能的权重
    """
    
    def __init__(self, project_root: Union[str, Path] = None):
        self.text_predictor = None
        self.audio_loader = None
        self.visual_predictor = None
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        
        # 模型路径 (使用验证可用的官方模型)
        self.model_paths = {
            'text': str(self.project_root / "06_模型文件" / "hmtl_models_v2" / "best_model_v2.pt"),
            'audio': str(self.project_root / "06_模型文件" / "audio_hmtl_v2_best.pt"),  # Wav2Vec2完整模型(75.4%)
            'visual': str(self.project_root / "06_模型文件" / "visual_hmtl_v4_best.pt")  # EfficientNet-B2 V4(62.9%)
        }
        
        self.available_modalities = []
        
        # 基于真实测试的模型权重 (使用新训练后的准确率)
        self.model_weights = {
            'text': 0.65,    # 自有文本真实场景估计约60-66%
            'audio': 0.754,  # Wav2Vec2音频模型记录准确率75.4%
            'visual': 0.629  # EfficientNet-B2 V4测试准确率62.9%
        }
        
        # 情绪标签映射
        self.emotion_labels = {
            '4_class': ['积极', '激活消极', '非激活消极', '平静'],
            '3_class': ['积极', '消极', '平静'],
            '7_class': ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        }
        
        # 4类到7类的映射关系
        self.class_4_to_7_mapping = {
            '积极': ['快乐', '支持'],
            '激活消极': ['愤怒', '焦虑'],
            '非激活消极': ['悲伤', '失望'],
            '平静': ['平静']
        }
    
    def initialize_models(self, load_visual=True, load_audio=True):
        """
        初始化所有模型
        Args:
            load_visual: 是否加载视觉模型
            load_audio: 是否加载音频模型
        """
        print("🔄 初始化多模态融合系统 V2...")
        print("="*60)
        self.available_modalities = []
        
        # 加载文本模型
        try:
            if os.path.exists(self.model_paths['text']):
                print("📝 加载文本模型...")
                self.text_predictor = HMTLPredictor(self.model_paths['text'])
                self.available_modalities.append('text')
                print("✅ 文本模型加载成功 (真实准确率: ~65%)")
        except Exception as e:
            print(f"❌ 文本模型加载失败: {e}")
        
        # 加载音频模型
        if load_audio:
            try:
                if os.path.exists(self.model_paths['audio']):
                    print("🎵 加载音频模型...")
                    self.audio_loader = AudioModelLoader(self.model_paths['audio'])
                    if self.audio_loader.load_model():
                        self.available_modalities.append('audio')
                        print("✅ 音频模型加载成功")
            except Exception as e:
                print(f"❌ 音频模型加载失败: {e}")
        
        # 加载视觉模型
        if load_visual and TORCHVISION_AVAILABLE:
            try:
                print("👁️ 加载视觉模型...")
                self.visual_predictor = RealVisualPredictor(
                    model_path=self.model_paths['visual'],
                    backbone='efficientnet_b2'
                )
                if self.visual_predictor.load_model():
                    self.available_modalities.append('visual')
                    print("✅ 视觉模型加载成功 (准确率: 62.9%)")
            except Exception as e:
                print(f"❌ 视觉模型加载失败: {e}")
        
        print(f"\n🎯 可用模态: {', '.join(self.available_modalities)}")
        print("="*60)
        
        return len(self.available_modalities) >= 1
    
    def _calculate_confidence(self, probs: torch.Tensor) -> float:
        """
        计算动态置信度
        基于softmax概率和熵
        """
        if probs is None:
            return 0.5
        
        max_prob = probs.max().item()
        
        # 计算熵 (越低越确定)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 综合置信度: 高概率 + 低熵 = 高置信度
        confidence = max_prob * (1 - 0.5 * normalized_entropy)
        return confidence
    
    def predict_text(self, text: str) -> Dict:
        """文本模态预测"""
        if 'text' not in self.available_modalities:
            return None
        
        try:
            result = self.text_predictor.predict(text, return_details=True)
            
            # 计算置信度
            probs_4 = torch.tensor(list(result.get('label_4_probs', {}).values()))
            confidence = self._calculate_confidence(probs_4) if len(probs_4) > 0 else 0.65
            
            return {
                '4类分类': result['emotion_4'],
                '3类极性': result['polarity_3'],
                '7类情绪': result['emotion_7'],
                'arousal': result['arousal'],
                'valence': result['valence'],
                'confidence': confidence,
                'modality': 'text'
            }
        except Exception as e:
            print(f"文本预测失败: {e}")
            return None
    
    def predict_audio(self, audio_input: Union[str, np.ndarray]) -> Dict:
        """
        音频模态预测
        Args:
            audio_input: 音频文件路径或音频数组
        """
        if 'audio' not in self.available_modalities:
            return None
        
        try:
            result = self.audio_loader.predict_from_audio(audio_input)
            
            return {
                '4类分类': result['4类分类'],
                '3类极性': result['3类极性'],
                '7类情绪': result['7类情绪'],
                'arousal': result['arousal'],
                'valence': result['valence'],
                'confidence': result.get('confidence', 0.60),
                'modality': 'audio'
            }
        except Exception as e:
            print(f"音频预测失败: {e}")
            return None
    
    def predict_visual(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict:
        """
        视觉模态预测
        Args:
            image_input: 图像文件路径、numpy数组或PIL Image
        """
        if 'visual' not in self.available_modalities:
            return None
        
        try:
            result = self.visual_predictor.predict_from_image(image_input)
            
            return {
                '4类分类': result['4类分类'],
                '3类极性': result['3类极性'],
                '7类情绪': result['7类情绪'],
                'arousal': result['arousal'],
                'valence': result['valence'],
                'confidence': result['confidence'],
                'modality': 'visual'
            }
        except Exception as e:
            print(f"视觉预测失败: {e}")
            return None
    
    def get_predictions(self, 
                       text: str = None,
                       audio: Union[str, np.ndarray] = None,
                       image: Union[str, np.ndarray, Image.Image] = None) -> Dict:
        """
        获取各模态的预测结果
        只处理提供的输入，不使用随机数据
        """
        predictions = {}
        
        if text is not None:
            text_pred = self.predict_text(text)
            if text_pred:
                predictions['text'] = text_pred
        
        if audio is not None:
            audio_pred = self.predict_audio(audio)
            if audio_pred:
                predictions['audio'] = audio_pred
        
        if image is not None:
            visual_pred = self.predict_visual(image)
            if visual_pred:
                predictions['visual'] = visual_pred
        
        return predictions
    
    def weighted_voting_fusion(self, predictions: Dict) -> Dict:
        """加权投票融合"""
        if not predictions:
            return None
        
        # 4类分类加权投票
        weighted_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4类分类']
            # 使用动态权重: 基础权重 * 置信度
            weight = self.model_weights.get(modality, 0.5) * pred.get('confidence', 0.5)
            weighted_votes_4[class_4] = weighted_votes_4.get(class_4, 0) + weight
        
        # 3类极性加权投票
        weighted_votes_3 = {}
        for modality, pred in predictions.items():
            class_3 = pred['3类极性']
            weight = self.model_weights.get(modality, 0.5) * pred.get('confidence', 0.5)
            weighted_votes_3[class_3] = weighted_votes_3.get(class_3, 0) + weight
        
        # 7类情绪加权投票
        weighted_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            weight = self.model_weights.get(modality, 0.5) * pred.get('confidence', 0.5)
            weighted_votes_7[class_7] = weighted_votes_7.get(class_7, 0) + weight
        
        # 加权平均 Arousal/Valence
        total_weight = sum(
            self.model_weights.get(mod, 0.5) * pred.get('confidence', 0.5)
            for mod, pred in predictions.items()
        )
        
        if total_weight > 0:
            arousal = sum(
                pred['arousal'] * self.model_weights.get(mod, 0.5) * pred.get('confidence', 0.5)
                for mod, pred in predictions.items()
            ) / total_weight
            
            valence = sum(
                pred['valence'] * self.model_weights.get(mod, 0.5) * pred.get('confidence', 0.5)
                for mod, pred in predictions.items()
            ) / total_weight
        else:
            arousal = np.mean([pred['arousal'] for pred in predictions.values()])
            valence = np.mean([pred['valence'] for pred in predictions.values()])
        
        return {
            '4类分类': max(weighted_votes_4, key=weighted_votes_4.get),
            '3类极性': max(weighted_votes_3, key=weighted_votes_3.get),
            '7类情绪': max(weighted_votes_7, key=weighted_votes_7.get),
            'arousal': round(arousal, 3),
            'valence': round(valence, 3),
            'strategy': 'weighted_voting',
            'vote_details': {
                '4类投票': weighted_votes_4,
                '7类投票': weighted_votes_7
            },
            'modalities_used': list(predictions.keys())
        }
    
    def hierarchical_fusion(self, predictions: Dict) -> Dict:
        """层次化融合 - 先确定4类，再在对应7类中选择"""
        if not predictions:
            return None
        
        # 第一层: 4类融合
        weighted_result = self.weighted_voting_fusion(predictions)
        primary_4class = weighted_result['4类分类']
        
        # 第二层: 在4类对应的7类候选中选择
        candidates = self.class_4_to_7_mapping.get(primary_4class, [])
        
        # 在候选中重新投票
        filtered_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            if class_7 in candidates:
                weight = self.model_weights.get(modality, 0.5) * pred.get('confidence', 0.5)
                filtered_votes_7[class_7] = filtered_votes_7.get(class_7, 0) + weight
        
        # 如果没有候选匹配，使用原始结果
        if filtered_votes_7:
            final_7class = max(filtered_votes_7, key=filtered_votes_7.get)
        else:
            final_7class = weighted_result['7类情绪']
        
        return {
            '4类分类': primary_4class,
            '3类极性': weighted_result['3类极性'],
            '7类情绪': final_7class,
            'arousal': weighted_result['arousal'],
            'valence': weighted_result['valence'],
            'strategy': 'hierarchical_fusion',
            'candidates': candidates,
            'modalities_used': list(predictions.keys())
        }
    
    def fuse(self, 
            text: str = None,
            audio: Union[str, np.ndarray] = None,
            image: Union[str, np.ndarray, Image.Image] = None,
            strategy: str = 'weighted') -> Dict:
        """
        执行多模态融合
        Args:
            text: 文本输入
            audio: 音频输入 (文件路径或数组)
            image: 图像输入 (文件路径、数组或PIL Image)
            strategy: 融合策略 ('weighted' 或 'hierarchical')
        """
        # 获取各模态预测
        predictions = self.get_predictions(text=text, audio=audio, image=image)
        
        if not predictions:
            print("⚠️ 没有可用的预测结果")
            return None
        
        if len(predictions) == 1:
            # 只有一个模态，直接返回
            modality = list(predictions.keys())[0]
            result = predictions[modality].copy()
            result['strategy'] = 'single_modality'
            result['modalities_used'] = [modality]
            return {
                'single_predictions': predictions,
                'fusion_result': result
            }
        
        # 执行融合
        if strategy == 'hierarchical':
            fusion_result = self.hierarchical_fusion(predictions)
        else:
            fusion_result = self.weighted_voting_fusion(predictions)
        
        return {
            'single_predictions': predictions,
            'fusion_result': fusion_result
        }
    
    def print_result(self, result: Dict):
        """打印预测结果"""
        if not result:
            print("无结果")
            return
        
        print("\n📊 单模态预测结果:")
        for modality, pred in result.get('single_predictions', {}).items():
            print(f"  {modality.upper()}:")
            print(f"    4类分类: {pred['4类分类']}")
            print(f"    7类情绪: {pred['7类情绪']}")
            print(f"    置信度: {pred['confidence']:.3f}")
        
        fusion = result.get('fusion_result')
        if fusion:
            print(f"\n🔗 融合结果 ({fusion['strategy']}):")
            print(f"    4类分类: {fusion['4类分类']}")
            print(f"    7类情绪: {fusion['7类情绪']}")
            print(f"    Arousal: {fusion['arousal']}")
            print(f"    Valence: {fusion['valence']}")
            print(f"    使用模态: {fusion['modalities_used']}")


def test_fusion_system():
    """测试融合系统"""
    print("🧪 测试多模态融合系统 V2")
    print("="*60)
    
    # 初始化系统
    system = MultimodalFusionSystemV2()
    if not system.initialize_models(load_visual=True, load_audio=True):
        print("❌ 系统初始化失败")
        return
    
    # 测试1: 仅文本
    print("\n【测试1: 仅文本输入】")
    result = system.fuse(text="我今天心情很好，工作进展顺利")
    system.print_result(result)
    
    # 测试2: 文本 + 随机图像 (模拟)
    print("\n【测试2: 文本 + 图像输入】")
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = system.fuse(
        text="我很担心明天的考试",
        image=test_image
    )
    system.print_result(result)
    
    # 测试3: 文本 + 音频 (模拟)
    print("\n【测试3: 文本 + 音频输入】")
    test_audio = np.random.randn(16000)  # 1秒随机音频
    result = system.fuse(
        text="感觉很失落和沮丧",
        audio=test_audio
    )
    system.print_result(result)
    
    # 测试4: 三模态融合
    print("\n【测试4: 三模态融合】")
    result = system.fuse(
        text="心情很平静",
        audio=test_audio,
        image=test_image
    )
    system.print_result(result)
    
    print("\n✅ 融合系统测试完成！")
    print("⚠️ 注意: 音频和图像使用随机数据，真实应用需要真实输入")


if __name__ == "__main__":
    test_fusion_system()
