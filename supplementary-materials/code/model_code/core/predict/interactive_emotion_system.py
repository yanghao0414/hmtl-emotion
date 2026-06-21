#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式HMTL多模态情绪预测系统
支持模型选择和自定义测试
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
import numpy as np

class InteractiveEmotionSystem:
    """交互式多模态情绪预测系统"""
    
    def __init__(self):
        """初始化系统"""
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
        self.current_modality = None
        
    def initialize_system(self):
        """初始化所有模型"""
        print("🔄 初始化HMTL多模态情绪预测系统...")
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
        
        return len(self.available_modalities) > 0
    
    def show_model_menu(self):
        """显示模型选择菜单"""
        print("\n📋 模型选择菜单:")
        print("-" * 30)
        
        options = []
        if 'text' in self.available_modalities:
            options.append(('1', 'text', '📝 文本模型 (BERT-based, 99.37%准确率)'))
        if 'audio' in self.available_modalities:
            options.append(('2', 'audio', '🎵 音频模型 (Wav2Vec2-based)'))
        if 'visual' in self.available_modalities:
            options.append(('3', 'visual', '👁️ 视觉模型 (CNN-based, 63.8%准确率)'))
        
        if len(self.available_modalities) > 1:
            options.append(('4', 'fusion', '🔗 多模态融合预测'))
        
        options.append(('0', 'exit', '❌ 退出系统'))
        
        for num, key, desc in options:
            print(f"  {num}. {desc}")
        
        return options
    
    def predict_single_modality(self, modality, text_input):
        """单模态预测"""
        if modality == 'text' and 'text' in self.available_modalities:
            result = self.text_predictor.predict(text_input, return_details=False)
            return {
                '4类分类': result['emotion_4'],
                '3类极性': result['polarity_3'],
                '7类情绪': result['emotion_7'],
                'arousal': result['arousal'],
                'valence': result['valence'],
                '模态': 'text'
            }
        
        elif modality == 'audio' and 'audio' in self.available_modalities:
            # 生成测试音频数据
            test_audio = np.random.randn(16000)  # 1秒随机音频
            result = self.audio_loader.predict_from_audio(test_audio)
            result['模态'] = 'audio'
            return result
        
        elif modality == 'visual' and 'visual' in self.available_modalities:
            result = self.visual_predictor.predict_from_text_placeholder(text_input)
            result['模态'] = 'visual'
            return result
        
        return None
    
    def predict_fusion(self, text_input):
        """多模态融合预测"""
        predictions = []
        results = {}
        
        # 收集各模态预测
        if 'text' in self.available_modalities:
            text_result = self.predict_single_modality('text', text_input)
            if text_result:
                results['text'] = text_result
                predictions.append(text_result)
        
        if 'audio' in self.available_modalities:
            audio_result = self.predict_single_modality('audio', text_input)
            if audio_result:
                results['audio'] = audio_result
                predictions.append(audio_result)
        
        if 'visual' in self.available_modalities:
            visual_result = self.predict_single_modality('visual', text_input)
            if visual_result:
                results['visual'] = visual_result
                predictions.append(visual_result)
        
        if not predictions:
            return None
        
        # 简单投票融合
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
    
    def display_result(self, result, modality_name):
        """显示预测结果"""
        if result is None:
            print("❌ 预测失败")
            return
        
        print(f"\n📊 {modality_name}预测结果:")
        print("-" * 40)
        
        if modality_name == "多模态融合":
            # 显示各模态结果
            for mod, res in result.items():
                if mod != 'fusion':
                    print(f"  {mod.upper()}模态:")
                    print(f"    4类分类: {res['4类分类']}")
                    print(f"    7类情绪: {res['7类情绪']}")
            
            print(f"  融合结果:")
            fusion = result['fusion']
            print(f"    4类分类: {fusion['4类分类']}")
            print(f"    3类极性: {fusion['3类极性']}")
            print(f"    7类情绪: {fusion['7类情绪']}")
            print(f"    Arousal: {fusion['arousal']}")
            print(f"    Valence: {fusion['valence']}")
            print(f"    使用模态: {fusion['模态']}")
        else:
            # 单模态结果
            print(f"  4类分类: {result['4类分类']}")
            print(f"  3类极性: {result['3类极性']}")
            print(f"  7类情绪: {result['7类情绪']}")
            print(f"  Arousal: {result['arousal']}")
            print(f"  Valence: {result['valence']}")
    
    def run_interactive_mode(self):
        """运行交互式模式"""
        if not self.initialize_system():
            print("❌ 没有可用的模型，系统退出")
            return
        
        print("\n🎯 欢迎使用HMTL多模态情绪预测系统！")
        
        while True:
            try:
                options = self.show_model_menu()
                
                choice = input("\n请选择模型 (输入数字): ").strip()
                
                if choice == '0':
                    print("👋 感谢使用，再见！")
                    break
                
                # 找到选择的选项
                selected_option = None
                for num, key, desc in options:
                    if num == choice:
                        selected_option = (key, desc)
                        break
                
                if not selected_option:
                    print("❌ 无效选择，请重新输入")
                    continue
                
                modality, desc = selected_option
                
                if modality == 'exit':
                    print("👋 感谢使用，再见！")
                    break
                
                # 获取用户输入
                if modality == 'audio':
                    print(f"\n🎵 选择了音频模型")
                    print("注意: 音频模型将使用随机生成的测试音频数据")
                    text_input = input("请输入文本描述 (用于参考): ").strip()
                    if not text_input:
                        text_input = "测试音频"
                else:
                    print(f"\n{desc.split()[0]} 选择了{desc.split()[1]}")
                    text_input = input("请输入要分析的文本: ").strip()
                    if not text_input:
                        print("❌ 输入不能为空")
                        continue
                
                print(f"\n🔄 正在使用{desc.split()[1]}进行预测...")
                
                # 执行预测
                if modality == 'fusion':
                    result = self.predict_fusion(text_input)
                    self.display_result(result, "多模态融合")
                else:
                    result = self.predict_single_modality(modality, text_input)
                    modality_names = {
                        'text': '文本模型',
                        'audio': '音频模型', 
                        'visual': '视觉模型'
                    }
                    self.display_result(result, modality_names.get(modality, modality))
                
                # 询问是否继续
                continue_choice = input("\n是否继续测试? (y/n): ").strip().lower()
                if continue_choice in ['n', 'no', '否']:
                    print("👋 感谢使用，再见！")
                    break
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                continue


def main():
    """主函数"""
    system = InteractiveEmotionSystem()
    system.run_interactive_mode()


if __name__ == "__main__":
    main()
