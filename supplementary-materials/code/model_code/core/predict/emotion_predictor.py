#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情绪预测系统汇总 - 简单调用版本
直接调用即可获得三个分类结果
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
from hmtl_evaluate import HMTLPredictor

class EmotionPredictor:
    """情绪预测器 - 简化版本"""
    
    def __init__(self):
        """初始化预测器，使用最佳可用模型"""
        # 尝试多个模型路径，按优先级排序
        model_paths = [
            r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt",  # V1模型，架构兼容
            r"d:\bigcreate\06_模型文件\hmtl_models_v2\final_model_v2.pt", # V1备用
            r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt"  # V2模型
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise FileNotFoundError("找不到可用的模型文件")
        
        print(f"🔄 加载情绪预测模型: {os.path.basename(model_path)}")
        try:
            self.predictor = HMTLPredictor(model_path)
            print("✅ 预测系统就绪\n")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 如果第一个模型失败，尝试下一个
            for backup_path in model_paths[1:]:
                if os.path.exists(backup_path) and backup_path != model_path:
                    print(f"🔄 尝试备用模型: {os.path.basename(backup_path)}")
                    try:
                        self.predictor = HMTLPredictor(backup_path)
                        print("✅ 备用模型加载成功\n")
                        return
                    except Exception as backup_e:
                        print(f"❌ 备用模型也失败: {backup_e}")
                        continue
            raise Exception("所有模型都无法加载")
    
    def predict(self, text):
        """
        预测文本情绪 - 返回三个分类结果
        
        Args:
            text (str): 输入文本
            
        Returns:
            dict: {
                '4类分类': str,    # 积极/激活消极/非激活消极/平静
                '3类极性': str,    # 积极/消极/平静
                '7类情绪': str,    # 愤怒/焦虑/快乐/悲伤/失望/支持/平静
                'arousal': float,  # 唤醒度 0-1
                'valence': float   # 效价 -1到1
            }
        """
        result = self.predictor.predict(text, return_details=False)
        
        return {
            '4类分类': result['emotion_4'],
            '3类极性': result['polarity_3'], 
            '7类情绪': result['emotion_7'],
            'arousal': result['arousal'],
            'valence': result['valence']
        }
    
    def predict_batch(self, texts):
        """批量预测"""
        return [self.predict(text) for text in texts]
    
    def quick_classify(self, text):
        """快速获取4类分类结果"""
        return self.predict(text)['4类分类']


# 全局预测器实例
_predictor = None

def get_predictor():
    """获取预测器实例（单例模式）"""
    global _predictor
    if _predictor is None:
        _predictor = EmotionPredictor()
    return _predictor

def predict_emotion(text):
    """
    快速预测接口
    
    Args:
        text (str): 输入文本
        
    Returns:
        dict: 预测结果
    """
    return get_predictor().predict(text)

def classify_emotion(text):
    """
    快速分类接口 - 只返回4类分类
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 4类分类结果
    """
    return get_predictor().quick_classify(text)


def demo():
    """演示使用"""
    print("="*50)
    print("情绪预测系统演示")
    print("="*50)
    
    # 测试文本
    test_texts = [
        "我今天心情很好",
        "我很担心明天的考试",
        "感觉很失落",
        "心里很平静",
        "我非常愤怒！",
        "谢谢你的帮助"
    ]
    
    print("📊 预测结果:")
    print("-" * 50)
    
    for text in test_texts:
        result = predict_emotion(text)
        print(f"文本: {text}")
        print(f"  → 4类分类: {result['4类分类']}")
        print(f"  → 3类极性: {result['3类极性']}")
        print(f"  → 7类情绪: {result['7类情绪']}")
        print()
    
    print("🚀 快速分类:")
    print("-" * 50)
    for text in test_texts:
        classification = classify_emotion(text)
        print(f"{text:<15} → {classification}")


if __name__ == "__main__":
    demo()
