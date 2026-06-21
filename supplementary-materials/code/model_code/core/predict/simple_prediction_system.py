#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的HMTL情绪预测调用系统
快速调用三个分类预测
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

class SimplePredictionSystem:
    """简单预测系统 - 支持多模型选择"""
    
    def __init__(self, model_type="v2"):
        """
        初始化预测器
        
        Args:
            model_type (str): 模型类型 - "text", "audio", "visual"
        """
        self.available_models = {
            "v2": {
                "path": r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt",
                "description": "V2模型 - 99.37%准确率的HMTL情绪识别模型"
            },
            "final": {
                "path": r"d:\bigcreate\06_模型文件\hmtl_models_v2\final_model_v2.pt",
                "description": "Final模型 - V2架构的最终训练模型"
            },
            "text": {
                "path": r"d:\bigcreate\06_模型文件\text_hmtl_v3_best.pt",
                "description": "文本模型 - 基于BERT的文本情绪识别（可能架构不兼容）"
            }
        }
        
        self.current_model_type = None
        self.predictor = None
        
        # 加载指定模型
        self.load_model(model_type)
    
    def load_model(self, model_type):
        """
        加载指定类型的模型
        
        Args:
            model_type (str): 模型类型
        """
        if model_type not in self.available_models:
            raise ValueError(f"不支持的模型类型: {model_type}. 可选: {list(self.available_models.keys())}")
        
        model_info = self.available_models[model_type]
        model_path = model_info["path"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"🔄 加载{model_type}模型...")
        print(f"📁 路径: {model_path}")
        print(f"📝 描述: {model_info['description']}")
        
        try:
            self.predictor = HMTLPredictor(model_path)
            self.current_model_type = model_type
            print(f"✅ {model_type}模型加载完成\n")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("🔄 尝试加载备用模型...")
            
            # 尝试加载V2模型作为备用
            if model_type != "v2":
                backup_path = self.available_models["v2"]["path"]
                if os.path.exists(backup_path):
                    print(f"📁 备用路径: {backup_path}")
                    self.predictor = HMTLPredictor(backup_path)
                    self.current_model_type = "v2"
                    print(f"✅ 已加载备用V2模型\n")
                else:
                    raise FileNotFoundError("所有模型都无法加载")
    
    def switch_model(self, model_type):
        """
        切换到不同的模型
        
        Args:
            model_type (str): 要切换的模型类型
        """
        if model_type == self.current_model_type:
            print(f"⚠️  当前已经是{model_type}模型")
            return
        
        print(f"🔄 从{self.current_model_type}模型切换到{model_type}模型...")
        self.load_model(model_type)
    
    def list_available_models(self):
        """列出所有可用的模型"""
        print("📋 可用模型列表:")
        print("-" * 50)
        for model_type, info in self.available_models.items():
            status = "✅ 当前使用" if model_type == self.current_model_type else "⭕ 可切换"
            exists = "📁 存在" if os.path.exists(info["path"]) else "❌ 缺失"
            print(f"  {status} {model_type.upper():<8} - {info['description']} ({exists})")
        print()
    
    def get_current_model_info(self):
        """获取当前模型信息"""
        if not self.current_model_type:
            return "未加载模型"
        
        return {
            "type": self.current_model_type,
            "description": self.available_models[self.current_model_type]["description"],
            "path": self.available_models[self.current_model_type]["path"]
        }
    
    def predict_single(self, text):
        """
        预测单条文本的三个分类
        
        Args:
            text (str): 输入文本
            
        Returns:
            dict: 包含三个分类结果
        """
        result = self.predictor.predict(text, return_details=False)
        
        return {
            '文本': text,
            '4类分类': result['emotion_4'],      # 主要分类
            '3类极性': result['polarity_3'],     # 极性分类  
            '7类情绪': result['emotion_7'],      # 辅助情绪
            'Arousal': result['arousal'],        # 唤醒度
            'Valence': result['valence']         # 效价
        }
    
    def predict_batch(self, texts):
        """
        批量预测多条文本
        
        Args:
            texts (list): 文本列表
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def quick_classify(self, text):
        """
        快速分类 - 只返回主要的4类分类结果
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 4类分类结果
        """
        result = self.predictor.predict(text, return_details=False)
        return result['emotion_4']
    
    def get_emotion_intensity(self, text):
        """
        获取情绪强度描述
        
        Args:
            text (str): 输入文本
            
        Returns:
            dict: 包含强度描述的结果
        """
        result = self.predictor.predict(text, return_details=False)
        
        # 判断情绪强度
        arousal = result['arousal']
        if arousal > 0.7:
            intensity = "强烈"
        elif arousal > 0.4:
            intensity = "中等"
        else:
            intensity = "温和"
        
        return {
            '情绪': result['emotion_7'],
            '分类': result['emotion_4'],
            '强度': intensity,
            '描述': f"这是一种{intensity}的{result['emotion_7']}情绪"
        }


def demo():
    """演示系统使用"""
    print("="*60)
    print("HMTL多模型情绪预测系统演示")
    print("="*60)
    
    # 初始化系统 - 默认使用文本模型
    system = SimplePredictionSystem(model_type="text")
    
    # 显示可用模型
    system.list_available_models()
    
    # 测试文本
    test_texts = [
        "我今天心情很好",
        "我很担心明天的考试", 
        "感觉很失落和难过",
        "心里很平静",
        "我非常愤怒！",
        "谢谢你的帮助"
    ]
    
    print("1. 单条预测演示:")
    print("-" * 30)
    for text in test_texts[:3]:
        result = system.predict_single(text)
        print(f"文本: {result['文本']}")
        print(f"  → 4类分类: {result['4类分类']}")
        print(f"  → 3类极性: {result['3类极性']}")
        print(f"  → 7类情绪: {result['7类情绪']}")
        print()
    
    print("2. 批量预测演示:")
    print("-" * 30)
    batch_results = system.predict_batch(test_texts)
    for result in batch_results:
        print(f"{result['文本']:<12} → {result['4类分类']}")
    
    print("\n3. 快速分类演示:")
    print("-" * 30)
    for text in test_texts:
        classification = system.quick_classify(text)
        print(f"{text:<12} → {classification}")
    
    print("\n4. 情绪强度演示:")
    print("-" * 30)
    for text in test_texts[:3]:
        intensity_result = system.get_emotion_intensity(text)
        print(f"{text}: {intensity_result['描述']}")


def interactive_mode():
    """交互模式 - 支持模型切换"""
    print("\n" + "="*60)
    print("交互预测模式 (输入 'q' 退出)")
    print("特殊命令:")
    print("  'models' - 查看可用模型")
    print("  'switch text/audio/visual' - 切换模型")
    print("  'info' - 查看当前模型信息")
    print("="*60)
    
    system = SimplePredictionSystem(model_type="text")
    
    while True:
        try:
            current_model = system.current_model_type.upper()
            user_input = input(f"\n[{current_model}] 请输入文本或命令: ").strip()
            
            if user_input.lower() == 'q':
                print("再见！")
                break
            
            if not user_input:
                continue
            
            # 处理特殊命令
            if user_input.lower() == 'models':
                system.list_available_models()
                continue
            
            if user_input.lower() == 'info':
                info = system.get_current_model_info()
                print(f"\n📋 当前模型信息:")
                print(f"  类型: {info['type'].upper()}")
                print(f"  描述: {info['description']}")
                continue
            
            if user_input.lower().startswith('switch '):
                model_type = user_input.lower().replace('switch ', '').strip()
                try:
                    system.switch_model(model_type)
                except Exception as e:
                    print(f"❌ 切换失败: {e}")
                continue
            
            # 完整预测
            result = system.predict_single(user_input)
            intensity = system.get_emotion_intensity(user_input)
            
            print(f"\n📊 分析结果 (使用{current_model}模型):")
            print(f"  🎯 4类分类: {result['4类分类']}")
            print(f"  ⚖️  3类极性: {result['3类极性']}")
            print(f"  😊 7类情绪: {result['7类情绪']}")
            print(f"  📈 唤醒度: {result['Arousal']:.3f}")
            print(f"  💫 效价: {result['Valence']:+.3f}")
            print(f"  💡 {intensity['描述']}")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    # 运行演示
    demo()
    
    # 交互模式
    interactive_mode()
