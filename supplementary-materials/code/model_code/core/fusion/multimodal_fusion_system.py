#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合机制系统
专门用于HMTL情绪识别的多模态融合策略研究和实现
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
import torch.nn as nn
import numpy as np
from hmtl_evaluate import HMTLPredictor
from audio_model_loader import AudioModelLoader
from visual_model_simple import SimpleVisualPredictor
from typing import Dict, List, Tuple, Optional
import json

class MultimodalFusionSystem:
    """多模态融合系统"""
    
    def __init__(self):
        """初始化融合系统"""
        self.text_predictor = None
        self.audio_loader = None
        self.visual_predictor = None
        
        # 模型路径
        self.model_paths = {
            'text': r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt",
            'audio': r"d:\bigcreate\06_模型文件\audio_hmtl_v2_best.pt",
            'visual': r"d:\bigcreate\06_模型文件\visual_hmtl_v4_best.pt"
        }
        
        self.available_modalities = []
        
        # 融合策略配置
        self.fusion_strategies = {
            'simple_voting': self._simple_voting_fusion,
            'weighted_voting': self._weighted_voting_fusion,
            'confidence_weighted': self._confidence_weighted_fusion,
            'attention_fusion': self._attention_fusion,
            'hierarchical_fusion': self._hierarchical_fusion
        }
        
        # 模型性能权重 (基于已知准确率)
        self.model_weights = {
            'text': 0.99,    # 99%准确率
            'audio': 0.65,   # 估计65%准确率
            'visual': 0.64   # 64%准确率
        }
        
        # 情绪标签映射
        self.emotion_labels = {
            '4_class': ['积极', '激活消极', '非激活消极', '平静'],
            '3_class': ['积极', '消极', '平静'],
            '7_class': ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
        }
    
    def initialize_models(self):
        """初始化所有模型"""
        print("🔄 初始化多模态融合系统...")
        print("="*60)
        
        # 加载文本模型
        try:
            if os.path.exists(self.model_paths['text']):
                print("📝 加载文本模型...")
                self.text_predictor = HMTLPredictor(self.model_paths['text'])
                self.available_modalities.append('text')
                print("✅ 文本模型加载成功")
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
        except Exception as e:
            print(f"❌ 视觉模型加载失败: {e}")
        
        print(f"\n🎯 可用模态: {', '.join(self.available_modalities)}")
        print("="*60)
        
        return len(self.available_modalities) > 1
    
    def get_single_predictions(self, text_input: str) -> Dict:
        """获取各个模态的单独预测结果"""
        predictions = {}
        
        # 文本预测
        if 'text' in self.available_modalities:
            try:
                result = self.text_predictor.predict(text_input, return_details=False)
                predictions['text'] = {
                    '4类分类': result['emotion_4'],
                    '3类极性': result['polarity_3'],
                    '7类情绪': result['emotion_7'],
                    'arousal': result['arousal'],
                    'valence': result['valence'],
                    'confidence': 0.99  # 基于模型准确率
                }
            except Exception as e:
                print(f"文本预测失败: {e}")
        
        # 音频预测
        if 'audio' in self.available_modalities:
            try:
                test_audio = np.random.randn(16000)  # 测试音频
                result = self.audio_loader.predict_from_audio(test_audio)
                predictions['audio'] = {
                    '4类分类': result['4类分类'],
                    '3类极性': result['3类极性'],
                    '7类情绪': result['7类情绪'],
                    'arousal': result['arousal'],
                    'valence': result['valence'],
                    'confidence': 0.65  # 估计置信度
                }
            except Exception as e:
                print(f"音频预测失败: {e}")
        
        # 视觉预测
        if 'visual' in self.available_modalities:
            try:
                result = self.visual_predictor.predict_from_text_placeholder(text_input)
                predictions['visual'] = {
                    '4类分类': result['4类分类'],
                    '3类极性': result['3类极性'],
                    '7类情绪': result['7类情绪'],
                    'arousal': result['arousal'],
                    'valence': result['valence'],
                    'confidence': 0.64  # 基于模型准确率
                }
            except Exception as e:
                print(f"视觉预测失败: {e}")
        
        return predictions
    
    def _simple_voting_fusion(self, predictions: Dict) -> Dict:
        """简单投票融合策略"""
        if not predictions:
            return None
        
        # 4类分类投票
        votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4类分类']
            votes_4[class_4] = votes_4.get(class_4, 0) + 1
        
        # 3类极性投票
        votes_3 = {}
        for modality, pred in predictions.items():
            class_3 = pred['3类极性']
            votes_3[class_3] = votes_3.get(class_3, 0) + 1
        
        # 7类情绪投票
        votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            votes_7[class_7] = votes_7.get(class_7, 0) + 1
        
        # Arousal和Valence平均
        arousal_avg = np.mean([pred['arousal'] for pred in predictions.values()])
        valence_avg = np.mean([pred['valence'] for pred in predictions.values()])
        
        return {
            '4类分类': max(votes_4, key=votes_4.get),
            '3类极性': max(votes_3, key=votes_3.get),
            '7类情绪': max(votes_7, key=votes_7.get),
            'arousal': round(arousal_avg, 3),
            'valence': round(valence_avg, 3),
            'strategy': 'simple_voting',
            'modalities_used': list(predictions.keys())
        }
    
    def _weighted_voting_fusion(self, predictions: Dict) -> Dict:
        """基于模型性能的加权投票融合"""
        if not predictions:
            return None
        
        # 4类分类加权投票
        weighted_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4类分类']
            weight = self.model_weights.get(modality, 0.5)
            weighted_votes_4[class_4] = weighted_votes_4.get(class_4, 0) + weight
        
        # 3类极性加权投票
        weighted_votes_3 = {}
        for modality, pred in predictions.items():
            class_3 = pred['3类极性']
            weight = self.model_weights.get(modality, 0.5)
            weighted_votes_3[class_3] = weighted_votes_3.get(class_3, 0) + weight
        
        # 7类情绪加权投票
        weighted_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            weight = self.model_weights.get(modality, 0.5)
            weighted_votes_7[class_7] = weighted_votes_7.get(class_7, 0) + weight
        
        # Arousal和Valence加权平均
        total_weight = sum(self.model_weights.get(mod, 0.5) for mod in predictions.keys())
        arousal_weighted = sum(pred['arousal'] * self.model_weights.get(mod, 0.5) 
                              for mod, pred in predictions.items()) / total_weight
        valence_weighted = sum(pred['valence'] * self.model_weights.get(mod, 0.5) 
                              for mod, pred in predictions.items()) / total_weight
        
        return {
            '4类分类': max(weighted_votes_4, key=weighted_votes_4.get),
            '3类极性': max(weighted_votes_3, key=weighted_votes_3.get),
            '7类情绪': max(weighted_votes_7, key=weighted_votes_7.get),
            'arousal': round(arousal_weighted, 3),
            'valence': round(valence_weighted, 3),
            'strategy': 'weighted_voting',
            'weights_used': {mod: self.model_weights.get(mod, 0.5) for mod in predictions.keys()},
            'modalities_used': list(predictions.keys())
        }
    
    def _confidence_weighted_fusion(self, predictions: Dict) -> Dict:
        """基于置信度的动态加权融合"""
        if not predictions:
            return None
        
        # 计算动态权重 (置信度 * 模型性能)
        dynamic_weights = {}
        for modality, pred in predictions.items():
            base_weight = self.model_weights.get(modality, 0.5)
            confidence = pred.get('confidence', 0.5)
            dynamic_weights[modality] = base_weight * confidence
        
        # 归一化权重
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        # 4类分类置信度加权
        conf_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4类分类']
            weight = dynamic_weights.get(modality, 0)
            conf_votes_4[class_4] = conf_votes_4.get(class_4, 0) + weight
        
        # 3类极性置信度加权
        conf_votes_3 = {}
        for modality, pred in predictions.items():
            class_3 = pred['3类极性']
            weight = dynamic_weights.get(modality, 0)
            conf_votes_3[class_3] = conf_votes_3.get(class_3, 0) + weight
        
        # 7类情绪置信度加权
        conf_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            weight = dynamic_weights.get(modality, 0)
            conf_votes_7[class_7] = conf_votes_7.get(class_7, 0) + weight
        
        # Arousal和Valence置信度加权平均
        arousal_conf = sum(pred['arousal'] * dynamic_weights.get(mod, 0) 
                          for mod, pred in predictions.items())
        valence_conf = sum(pred['valence'] * dynamic_weights.get(mod, 0) 
                          for mod, pred in predictions.items())
        
        return {
            '4类分类': max(conf_votes_4, key=conf_votes_4.get),
            '3类极性': max(conf_votes_3, key=conf_votes_3.get),
            '7类情绪': max(conf_votes_7, key=conf_votes_7.get),
            'arousal': round(arousal_conf, 3),
            'valence': round(valence_conf, 3),
            'strategy': 'confidence_weighted',
            'dynamic_weights': dynamic_weights,
            'modalities_used': list(predictions.keys())
        }
    
    def _attention_fusion(self, predictions: Dict) -> Dict:
        """注意力机制融合 (简化版)"""
        if not predictions:
            return None
        
        # 基于情绪一致性计算注意力权重
        attention_weights = {}
        
        # 计算各模态预测的一致性分数
        for modality in predictions.keys():
            consistency_score = 0
            pred = predictions[modality]
            
            # 检查4类和7类的一致性
            class_4 = pred['4类分类']
            class_7 = pred['7类情绪']
            
            # 定义4类和7类的映射关系
            class_mapping = {
                '积极': ['快乐', '支持'],
                '激活消极': ['愤怒', '焦虑'],
                '非激活消极': ['悲伤', '失望'],
                '平静': ['平静']
            }
            
            if class_7 in class_mapping.get(class_4, []):
                consistency_score += 0.5
            
            # 检查arousal/valence与分类的一致性
            arousal = pred['arousal']
            valence = pred['valence']
            
            if class_4 == '积极' and valence > 0:
                consistency_score += 0.3
            elif class_4 == '激活消极' and arousal > 0.6 and valence < 0:
                consistency_score += 0.3
            elif class_4 == '非激活消极' and arousal < 0.4 and valence < 0:
                consistency_score += 0.3
            elif class_4 == '平静' and abs(valence) < 0.3:
                consistency_score += 0.3
            
            # 基础权重
            base_weight = self.model_weights.get(modality, 0.5)
            attention_weights[modality] = base_weight * (1 + consistency_score)
        
        # 归一化注意力权重
        total_attention = sum(attention_weights.values())
        if total_attention > 0:
            attention_weights = {k: v/total_attention for k, v in attention_weights.items()}
        
        # 使用注意力权重进行融合
        att_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4类分类']
            weight = attention_weights.get(modality, 0)
            att_votes_4[class_4] = att_votes_4.get(class_4, 0) + weight
        
        att_votes_3 = {}
        for modality, pred in predictions.items():
            class_3 = pred['3类极性']
            weight = attention_weights.get(modality, 0)
            att_votes_3[class_3] = att_votes_3.get(class_3, 0) + weight
        
        att_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            weight = attention_weights.get(modality, 0)
            att_votes_7[class_7] = att_votes_7.get(class_7, 0) + weight
        
        arousal_att = sum(pred['arousal'] * attention_weights.get(mod, 0) 
                         for mod, pred in predictions.items())
        valence_att = sum(pred['valence'] * attention_weights.get(mod, 0) 
                         for mod, pred in predictions.items())
        
        return {
            '4类分类': max(att_votes_4, key=att_votes_4.get),
            '3类极性': max(att_votes_3, key=att_votes_3.get),
            '7类情绪': max(att_votes_7, key=att_votes_7.get),
            'arousal': round(arousal_att, 3),
            'valence': round(valence_att, 3),
            'strategy': 'attention_fusion',
            'attention_weights': attention_weights,
            'modalities_used': list(predictions.keys())
        }
    
    def _hierarchical_fusion(self, predictions: Dict) -> Dict:
        """层次化融合策略"""
        if not predictions:
            return None
        
        # 第一层：基于4类主分类的融合
        primary_fusion = self._weighted_voting_fusion(predictions)
        
        # 第二层：基于主分类结果调整7类细分
        primary_4class = primary_fusion['4类分类']
        
        # 根据4类结果筛选7类候选
        class_7_candidates = {
            '积极': ['快乐', '支持'],
            '激活消极': ['愤怒', '焦虑'],
            '非激活消极': ['悲伤', '失望'],
            '平静': ['平静']
        }
        
        candidates = class_7_candidates.get(primary_4class, [])
        
        # 在候选中重新投票
        filtered_votes_7 = {}
        for modality, pred in predictions.items():
            class_7 = pred['7类情绪']
            if class_7 in candidates:
                weight = self.model_weights.get(modality, 0.5)
                filtered_votes_7[class_7] = filtered_votes_7.get(class_7, 0) + weight
        
        # 如果没有候选，使用原始结果
        if not filtered_votes_7:
            final_7class = primary_fusion['7类情绪']
        else:
            final_7class = max(filtered_votes_7, key=filtered_votes_7.get)
        
        return {
            '4类分类': primary_4class,
            '3类极性': primary_fusion['3类极性'],
            '7类情绪': final_7class,
            'arousal': primary_fusion['arousal'],
            'valence': primary_fusion['valence'],
            'strategy': 'hierarchical_fusion',
            'primary_result': primary_4class,
            'filtered_candidates': candidates,
            'modalities_used': list(predictions.keys())
        }
    
    def fuse_predictions(self, text_input: str, strategy: str = 'all') -> Dict:
        """执行多模态融合预测"""
        # 获取单模态预测
        single_predictions = self.get_single_predictions(text_input)
        
        if len(single_predictions) < 2:
            print("⚠️ 需要至少2个模态才能进行融合")
            return single_predictions
        
        fusion_results = {}
        
        if strategy == 'all':
            # 执行所有融合策略
            for strategy_name, fusion_func in self.fusion_strategies.items():
                try:
                    result = fusion_func(single_predictions)
                    if result:
                        fusion_results[strategy_name] = result
                except Exception as e:
                    print(f"融合策略 {strategy_name} 失败: {e}")
        else:
            # 执行指定策略
            if strategy in self.fusion_strategies:
                try:
                    result = self.fusion_strategies[strategy](single_predictions)
                    if result:
                        fusion_results[strategy] = result
                except Exception as e:
                    print(f"融合策略 {strategy} 失败: {e}")
        
        return {
            'single_predictions': single_predictions,
            'fusion_results': fusion_results,
            'input_text': text_input
        }
    
    def compare_fusion_strategies(self, text_input: str) -> None:
        """比较不同融合策略的结果"""
        results = self.fuse_predictions(text_input, 'all')
        
        print(f"\n🔍 融合策略对比分析")
        print(f"输入文本: {text_input}")
        print("="*80)
        
        # 显示单模态结果
        print("\n📊 单模态预测结果:")
        for modality, pred in results['single_predictions'].items():
            print(f"  {modality.upper()}:")
            print(f"    4类分类: {pred['4类分类']}")
            print(f"    7类情绪: {pred['7类情绪']}")
            print(f"    置信度: {pred['confidence']}")
        
        # 显示融合结果
        print(f"\n🔗 融合策略结果:")
        for strategy, result in results['fusion_results'].items():
            print(f"  {strategy.upper()}:")
            print(f"    4类分类: {result['4类分类']}")
            print(f"    7类情绪: {result['7类情绪']}")
            print(f"    Arousal: {result['arousal']}")
            print(f"    Valence: {result['valence']}")
            
            # 显示策略特定信息
            if 'weights_used' in result:
                print(f"    权重: {result['weights_used']}")
            elif 'attention_weights' in result:
                print(f"    注意力权重: {result['attention_weights']}")
            elif 'dynamic_weights' in result:
                print(f"    动态权重: {result['dynamic_weights']}")
            print()


def main():
    """主函数 - 融合机制测试"""
    fusion_system = MultimodalFusionSystem()
    
    if not fusion_system.initialize_models():
        print("❌ 模型初始化失败")
        return
    
    print("\n🎯 多模态融合机制测试")
    print("="*60)
    
    # 测试用例
    test_cases = [
        "我今天心情很好，工作进展顺利",
        "我很担心明天的考试",
        "感觉很失落和沮丧",
        "心情很平静"
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n【测试用例 {i}】")
        fusion_system.compare_fusion_strategies(text)
        
        if i < len(test_cases):
            input("\n按回车键继续下一个测试用例...")


if __name__ == "__main__":
    main()
