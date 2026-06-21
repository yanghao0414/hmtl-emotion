#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合评估框架
全面评估多模态融合系统的性能
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

import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter

from multimodal_fusion_v2 import MultimodalFusionSystemV2


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.fusion_system = None
        self.results = {}
        self.test_cases = []
        
        # 标准测试用例
        self.standard_test_cases = [
            # 积极情绪
            {"text": "我今天心情很好，工作进展顺利", "expected_4": "积极", "expected_7": "支持", "category": "positive"},
            {"text": "太开心了，终于完成了这个项目", "expected_4": "积极", "expected_7": "快乐", "category": "positive"},
            {"text": "感谢大家的支持，我很感动", "expected_4": "积极", "expected_7": "支持", "category": "positive"},
            {"text": "这个消息让我非常高兴", "expected_4": "积极", "expected_7": "快乐", "category": "positive"},
            
            # 激活消极
            {"text": "我很担心明天的考试，压力很大", "expected_4": "激活消极", "expected_7": "焦虑", "category": "active_negative"},
            {"text": "这件事让我非常愤怒", "expected_4": "激活消极", "expected_7": "愤怒", "category": "active_negative"},
            {"text": "我对这个结果感到焦虑不安", "expected_4": "激活消极", "expected_7": "焦虑", "category": "active_negative"},
            {"text": "他的行为让我很生气", "expected_4": "激活消极", "expected_7": "愤怒", "category": "active_negative"},
            
            # 非激活消极
            {"text": "感觉很失落和沮丧，什么都不想做", "expected_4": "非激活消极", "expected_7": "悲伤", "category": "passive_negative"},
            {"text": "我很难过，事情没有按预期发展", "expected_4": "非激活消极", "expected_7": "悲伤", "category": "passive_negative"},
            {"text": "对这个结果很失望", "expected_4": "非激活消极", "expected_7": "失望", "category": "passive_negative"},
            {"text": "心里很难受，感觉被辜负了", "expected_4": "非激活消极", "expected_7": "失望", "category": "passive_negative"},
            
            # 平静
            {"text": "心情很平静，没有什么特别的感受", "expected_4": "平静", "expected_7": "平静", "category": "neutral"},
            {"text": "今天是普通的一天", "expected_4": "平静", "expected_7": "平静", "category": "neutral"},
            {"text": "感觉还好，没什么特别的", "expected_4": "平静", "expected_7": "平静", "category": "neutral"},
            {"text": "一切都很正常", "expected_4": "平静", "expected_7": "平静", "category": "neutral"},
        ]
    
    def initialize(self, load_visual=True, load_audio=True):
        """初始化评估系统"""
        print("🔄 初始化综合评估系统...")
        
        self.fusion_system = MultimodalFusionSystemV2()
        success = self.fusion_system.initialize_models(
            load_visual=load_visual,
            load_audio=load_audio
        )
        
        if success:
            print("✅ 评估系统初始化成功")
        else:
            print("❌ 评估系统初始化失败")
        
        return success
    
    def evaluate_text_model(self, test_cases: List[Dict] = None) -> Dict:
        """评估文本模型"""
        print("\n📝 评估文本模型...")
        print("-"*50)
        
        if test_cases is None:
            test_cases = self.standard_test_cases
        
        correct_4 = 0
        correct_7 = 0
        total = 0
        predictions = []
        confusion_4 = []
        confusion_7 = []
        
        for case in test_cases:
            try:
                result = self.fusion_system.predict_text(case['text'])
                
                if result:
                    pred_4 = result['4类分类']
                    pred_7 = result['7类情绪']
                    expected_4 = case['expected_4']
                    expected_7 = case['expected_7']
                    
                    is_correct_4 = self._normalize_label(pred_4) == self._normalize_label(expected_4)
                    is_correct_7 = pred_7 == expected_7
                    
                    if is_correct_4:
                        correct_4 += 1
                    else:
                        confusion_4.append(f"{expected_4} → {pred_4}")
                    
                    if is_correct_7:
                        correct_7 += 1
                    else:
                        confusion_7.append(f"{expected_7} → {pred_7}")
                    
                    predictions.append({
                        'text': case['text'][:30] + '...',
                        'expected_4': expected_4,
                        'predicted_4': pred_4,
                        'correct_4': is_correct_4,
                        'expected_7': expected_7,
                        'predicted_7': pred_7,
                        'correct_7': is_correct_7,
                        'confidence': result['confidence']
                    })
                    
                    total += 1
                    
            except Exception as e:
                print(f"  ❌ 预测失败: {e}")
        
        accuracy_4 = correct_4 / total if total > 0 else 0
        accuracy_7 = correct_7 / total if total > 0 else 0
        
        result = {
            'accuracy_4': accuracy_4,
            'accuracy_7': accuracy_7,
            'correct_4': correct_4,
            'correct_7': correct_7,
            'total': total,
            'predictions': predictions,
            'confusion_4': Counter(confusion_4).most_common(5),
            'confusion_7': Counter(confusion_7).most_common(5)
        }
        
        print(f"  4类准确率: {accuracy_4:.2%} ({correct_4}/{total})")
        print(f"  7类准确率: {accuracy_7:.2%} ({correct_7}/{total})")
        
        if confusion_4:
            print(f"  4类主要混淆: {confusion_4[:3]}")
        
        self.results['text_model'] = result
        return result
    
    def _normalize_label(self, label: str) -> str:
        """标准化标签名称"""
        label_mapping = {
            '激活型消极': '激活消极',
            '非激活型消极': '非激活消极',
        }
        return label_mapping.get(label, label)
    
    def evaluate_fusion_strategies(self, test_cases: List[Dict] = None) -> Dict:
        """评估不同融合策略"""
        print("\n🔗 评估融合策略...")
        print("-"*50)
        
        if test_cases is None:
            test_cases = self.standard_test_cases
        
        strategies = ['weighted', 'hierarchical']
        strategy_results = {}
        
        for strategy in strategies:
            correct_4 = 0
            total = 0
            
            for case in test_cases:
                try:
                    # 仅使用文本（因为没有真实的音频/图像数据）
                    result = self.fusion_system.fuse(
                        text=case['text'],
                        strategy=strategy
                    )
                    
                    if result and 'fusion_result' in result:
                        pred_4 = result['fusion_result']['4类分类']
                        expected_4 = case['expected_4']
                        
                        if self._normalize_label(pred_4) == self._normalize_label(expected_4):
                            correct_4 += 1
                        
                        total += 1
                        
                except Exception as e:
                    print(f"  ❌ {strategy} 策略失败: {e}")
            
            accuracy = correct_4 / total if total > 0 else 0
            strategy_results[strategy] = {
                'accuracy_4': accuracy,
                'correct': correct_4,
                'total': total
            }
            
            print(f"  {strategy}: {accuracy:.2%} ({correct_4}/{total})")
        
        self.results['fusion_strategies'] = strategy_results
        return strategy_results
    
    def evaluate_by_category(self, test_cases: List[Dict] = None) -> Dict:
        """按类别评估"""
        print("\n📊 按类别评估...")
        print("-"*50)
        
        if test_cases is None:
            test_cases = self.standard_test_cases
        
        categories = {}
        
        for case in test_cases:
            category = case.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'correct': 0, 'total': 0}
            
            try:
                result = self.fusion_system.predict_text(case['text'])
                
                if result:
                    pred_4 = result['4类分类']
                    expected_4 = case['expected_4']
                    
                    categories[category]['total'] += 1
                    if self._normalize_label(pred_4) == self._normalize_label(expected_4):
                        categories[category]['correct'] += 1
                        
            except Exception as e:
                pass
        
        category_results = {}
        for category, counts in categories.items():
            accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            category_results[category] = {
                'accuracy': accuracy,
                'correct': counts['correct'],
                'total': counts['total']
            }
            print(f"  {category}: {accuracy:.2%} ({counts['correct']}/{counts['total']})")
        
        self.results['by_category'] = category_results
        return category_results
    
    def evaluate_confidence_calibration(self, test_cases: List[Dict] = None) -> Dict:
        """评估置信度校准"""
        print("\n📈 评估置信度校准...")
        print("-"*50)
        
        if test_cases is None:
            test_cases = self.standard_test_cases
        
        # 按置信度分组
        confidence_bins = {
            'high (>0.8)': {'correct': 0, 'total': 0},
            'medium (0.5-0.8)': {'correct': 0, 'total': 0},
            'low (<0.5)': {'correct': 0, 'total': 0}
        }
        
        for case in test_cases:
            try:
                result = self.fusion_system.predict_text(case['text'])
                
                if result:
                    confidence = result['confidence']
                    pred_4 = result['4类分类']
                    expected_4 = case['expected_4']
                    is_correct = self._normalize_label(pred_4) == self._normalize_label(expected_4)
                    
                    if confidence > 0.8:
                        bin_name = 'high (>0.8)'
                    elif confidence > 0.5:
                        bin_name = 'medium (0.5-0.8)'
                    else:
                        bin_name = 'low (<0.5)'
                    
                    confidence_bins[bin_name]['total'] += 1
                    if is_correct:
                        confidence_bins[bin_name]['correct'] += 1
                        
            except Exception as e:
                pass
        
        calibration_results = {}
        for bin_name, counts in confidence_bins.items():
            accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
            calibration_results[bin_name] = {
                'accuracy': accuracy,
                'correct': counts['correct'],
                'total': counts['total']
            }
            print(f"  {bin_name}: {accuracy:.2%} ({counts['correct']}/{counts['total']})")
        
        self.results['confidence_calibration'] = calibration_results
        return calibration_results
    
    def generate_report(self) -> str:
        """生成评估报告"""
        print("\n📋 生成评估报告...")
        print("="*60)
        
        report = []
        report.append("# HMTL多模态情绪识别系统评估报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "="*50)
        
        # 文本模型评估
        if 'text_model' in self.results:
            r = self.results['text_model']
            report.append("\n## 1. 文本模型性能")
            report.append(f"- 4类准确率: {r['accuracy_4']:.2%} ({r['correct_4']}/{r['total']})")
            report.append(f"- 7类准确率: {r['accuracy_7']:.2%} ({r['correct_7']}/{r['total']})")
            if r['confusion_4']:
                report.append(f"- 4类主要混淆: {r['confusion_4']}")
        
        # 融合策略评估
        if 'fusion_strategies' in self.results:
            report.append("\n## 2. 融合策略性能")
            for strategy, r in self.results['fusion_strategies'].items():
                report.append(f"- {strategy}: {r['accuracy_4']:.2%}")
        
        # 按类别评估
        if 'by_category' in self.results:
            report.append("\n## 3. 按类别性能")
            for category, r in self.results['by_category'].items():
                report.append(f"- {category}: {r['accuracy']:.2%}")
        
        # 置信度校准
        if 'confidence_calibration' in self.results:
            report.append("\n## 4. 置信度校准")
            for bin_name, r in self.results['confidence_calibration'].items():
                report.append(f"- {bin_name}: {r['accuracy']:.2%}")
        
        # 总结
        report.append("\n## 5. 总结")
        if 'text_model' in self.results:
            acc = self.results['text_model']['accuracy_4']
            if acc >= 0.7:
                report.append("- 模型性能: 良好")
            elif acc >= 0.5:
                report.append("- 模型性能: 中等")
            else:
                report.append("- 模型性能: 需要改进")
        
        report.append("\n" + "="*50)
        
        report_text = '\n'.join(report)
        
        # 保存报告
        report_path = f"d:\\bigcreate\\evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 报告已保存: {report_path}")
        except Exception as e:
            print(f"⚠️ 保存报告失败: {e}")
        
        print(report_text)
        
        return report_text
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("🚀 开始完整系统评估")
        print("="*60)
        
        if not self.initialize(load_visual=False, load_audio=False):
            print("❌ 初始化失败，无法评估")
            return
        
        # 运行所有评估
        self.evaluate_text_model()
        self.evaluate_fusion_strategies()
        self.evaluate_by_category()
        self.evaluate_confidence_calibration()
        
        # 生成报告
        self.generate_report()
        
        print("\n✅ 完整评估完成！")


def main():
    """主函数"""
    evaluator = ComprehensiveEvaluator()
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
