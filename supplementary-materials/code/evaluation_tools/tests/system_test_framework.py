#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合系统测试框架
全面测试系统性能、稳定性、融合策略效果
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

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from multimodal_fusion_system import MultimodalFusionSystem
from advanced_fusion_system import AdvancedMultimodalFusion

class SystemTestFramework:
    """系统测试框架"""
    
    def __init__(self):
        self.basic_fusion = None
        self.advanced_fusion = None
        self.test_results = {}
        self.test_data = []
        
        # 测试数据集
        self.test_cases = [
            # 积极情绪测试
            {"text": "我今天心情很好，工作进展顺利", "expected_4": "积极", "category": "positive"},
            {"text": "太开心了，终于完成了这个项目", "expected_4": "积极", "category": "positive"},
            {"text": "感谢大家的支持，我很感动", "expected_4": "积极", "category": "positive"},
            
            # 激活消极情绪测试
            {"text": "我很担心明天的考试，压力很大", "expected_4": "激活消极", "category": "active_negative"},
            {"text": "这件事让我非常愤怒", "expected_4": "激活消极", "category": "active_negative"},
            {"text": "我对这个结果感到焦虑不安", "expected_4": "激活消极", "category": "active_negative"},
            
            # 非激活消极情绪测试
            {"text": "感觉很失落和沮丧，什么都不想做", "expected_4": "非激活消极", "category": "passive_negative"},
            {"text": "我很难过，事情没有按预期发展", "expected_4": "非激活消极", "category": "passive_negative"},
            {"text": "对这个结果很失望", "expected_4": "非激活消极", "category": "passive_negative"},
            
            # 平静情绪测试
            {"text": "心情很平静，没有什么特别的感受", "expected_4": "平静", "category": "neutral"},
            {"text": "今天是普通的一天", "expected_4": "平静", "category": "neutral"},
            {"text": "感觉还好，没什么特别的", "expected_4": "平静", "category": "neutral"},
            
            # 边界情况测试
            {"text": "我既开心又担心", "expected_4": None, "category": "mixed"},
            {"text": "心情复杂，说不清楚", "expected_4": None, "category": "ambiguous"},
            {"text": "今天天气不错", "expected_4": None, "category": "neutral_content"},
            
            # 极端情况测试
            {"text": "！！！太棒了！！！", "expected_4": "积极", "category": "extreme_positive"},
            {"text": "完全崩溃了，受不了了", "expected_4": "激活消极", "category": "extreme_negative"},
            {"text": "", "expected_4": None, "category": "empty"},
            {"text": "哈哈哈哈哈哈哈", "expected_4": "积极", "category": "repetitive"}
        ]
        
        # 融合策略列表
        self.fusion_strategies = [
            'simple_voting',
            'weighted_voting', 
            'confidence_weighted',
            'attention_fusion',
            'hierarchical_fusion'
        ]
        
        self.advanced_strategies = [
            'neural_fusion',
            'adaptive_fusion',
            'dynamic_fusion'
        ]
    
    def initialize_systems(self):
        """初始化测试系统"""
        print("🔄 初始化测试系统...")
        
        # 初始化基础融合系统
        try:
            self.basic_fusion = MultimodalFusionSystem()
            if self.basic_fusion.initialize_models():
                print("✅ 基础融合系统初始化成功")
            else:
                print("❌ 基础融合系统初始化失败")
                return False
        except Exception as e:
            print(f"❌ 基础融合系统初始化错误: {e}")
            return False
        
        # 初始化高级融合系统
        try:
            self.advanced_fusion = AdvancedMultimodalFusion()
            if self.advanced_fusion.initialize_models():
                print("✅ 高级融合系统初始化成功")
            else:
                print("❌ 高级融合系统初始化失败")
        except Exception as e:
            print(f"❌ 高级融合系统初始化错误: {e}")
        
        print("="*60)
        return True
    
    def test_basic_fusion_strategies(self):
        """测试基础融合策略"""
        print("\n📊 测试基础融合策略...")
        print("="*60)
        
        strategy_results = {}
        
        for strategy in self.fusion_strategies:
            print(f"\n🔍 测试策略: {strategy}")
            strategy_results[strategy] = {
                'correct': 0,
                'total': 0,
                'predictions': [],
                'execution_times': [],
                'errors': []
            }
            
            for i, test_case in enumerate(self.test_cases):
                if test_case['expected_4'] is None:  # 跳过无明确期望的测试
                    continue
                
                try:
                    start_time = time.time()
                    
                    # 获取单模态预测
                    single_predictions = self.basic_fusion.get_single_predictions(test_case['text'])
                    
                    # 执行特定融合策略
                    fusion_func = self.basic_fusion.fusion_strategies[strategy]
                    result = fusion_func(single_predictions)
                    
                    execution_time = time.time() - start_time
                    
                    if result:
                        predicted_4 = result['4类分类']
                        is_correct = predicted_4 == test_case['expected_4']
                        
                        strategy_results[strategy]['predictions'].append({
                            'text': test_case['text'],
                            'expected': test_case['expected_4'],
                            'predicted': predicted_4,
                            'correct': is_correct,
                            'category': test_case['category']
                        })
                        
                        strategy_results[strategy]['execution_times'].append(execution_time)
                        strategy_results[strategy]['total'] += 1
                        if is_correct:
                            strategy_results[strategy]['correct'] += 1
                    
                except Exception as e:
                    strategy_results[strategy]['errors'].append(str(e))
                    print(f"  ❌ 测试用例 {i+1} 失败: {e}")
            
            # 计算准确率
            if strategy_results[strategy]['total'] > 0:
                accuracy = strategy_results[strategy]['correct'] / strategy_results[strategy]['total']
                avg_time = np.mean(strategy_results[strategy]['execution_times'])
                print(f"  准确率: {accuracy:.3f} ({strategy_results[strategy]['correct']}/{strategy_results[strategy]['total']})")
                print(f"  平均执行时间: {avg_time:.4f}秒")
                print(f"  错误数: {len(strategy_results[strategy]['errors'])}")
        
        self.test_results['basic_fusion'] = strategy_results
        return strategy_results
    
    def test_advanced_fusion_strategies(self):
        """测试高级融合策略"""
        if not self.advanced_fusion:
            print("⚠️ 高级融合系统未初始化，跳过测试")
            return {}
        
        print("\n🧠 测试高级融合策略...")
        print("="*60)
        
        advanced_results = {}
        
        # 简化测试（只测试几个关键用例）
        key_test_cases = [
            {"text": "我今天心情很好，工作进展顺利", "expected_4": "积极", "category": "positive"},
            {"text": "我很担心明天的考试，压力很大", "expected_4": "激活消极", "category": "active_negative"},
            {"text": "感觉很失落和沮丧", "expected_4": "非激活消极", "category": "passive_negative"},
            {"text": "心情很平静", "expected_4": "平静", "category": "neutral"}
        ]
        
        for strategy in self.advanced_strategies:
            print(f"\n🔍 测试高级策略: {strategy}")
            advanced_results[strategy] = {
                'predictions': [],
                'execution_times': [],
                'errors': []
            }
            
            for test_case in key_test_cases:
                try:
                    start_time = time.time()
                    
                    # 提取特征
                    features, single_predictions = self.advanced_fusion.extract_features(test_case['text'])
                    
                    # 执行特定高级策略
                    if strategy == 'neural_fusion':
                        result = self.advanced_fusion.neural_network_fusion(features)
                    elif strategy == 'adaptive_fusion':
                        result = self.advanced_fusion.adaptive_fusion_strategy(features)
                    elif strategy == 'dynamic_fusion':
                        result = self.advanced_fusion.dynamic_fusion_strategy(features)
                    
                    execution_time = time.time() - start_time
                    
                    if result:
                        advanced_results[strategy]['predictions'].append({
                            'text': test_case['text'],
                            'expected': test_case['expected_4'],
                            'predicted': result['4类分类'],
                            'strategy_info': result.get('strategy', ''),
                            'category': test_case['category']
                        })
                        
                        advanced_results[strategy]['execution_times'].append(execution_time)
                
                except Exception as e:
                    advanced_results[strategy]['errors'].append(str(e))
                    print(f"  ❌ 高级策略测试失败: {e}")
            
            # 显示结果
            if advanced_results[strategy]['execution_times']:
                avg_time = np.mean(advanced_results[strategy]['execution_times'])
                print(f"  平均执行时间: {avg_time:.4f}秒")
                print(f"  成功预测数: {len(advanced_results[strategy]['predictions'])}")
                print(f"  错误数: {len(advanced_results[strategy]['errors'])}")
        
        self.test_results['advanced_fusion'] = advanced_results
        return advanced_results
    
    def test_system_stability(self):
        """测试系统稳定性"""
        print("\n🔒 测试系统稳定性...")
        print("="*60)
        
        stability_results = {
            'repeated_predictions': {},
            'concurrent_predictions': {},
            'memory_usage': {},
            'error_handling': {}
        }
        
        # 1. 重复预测一致性测试
        print("\n🔄 重复预测一致性测试...")
        test_text = "我今天心情很好"
        repeated_results = []
        
        for i in range(5):
            try:
                single_predictions = self.basic_fusion.get_single_predictions(test_text)
                result = self.basic_fusion.fusion_strategies['weighted_voting'](single_predictions)
                if result:
                    repeated_results.append(result['4类分类'])
            except Exception as e:
                print(f"  重复测试 {i+1} 失败: {e}")
        
        # 检查一致性
        if repeated_results:
            unique_results = set(repeated_results)
            consistency_rate = repeated_results.count(repeated_results[0]) / len(repeated_results)
            stability_results['repeated_predictions'] = {
                'results': repeated_results,
                'consistency_rate': consistency_rate,
                'unique_count': len(unique_results)
            }
            print(f"  一致性率: {consistency_rate:.3f}")
            print(f"  唯一结果数: {len(unique_results)}")
        
        # 2. 错误处理测试
        print("\n⚠️ 错误处理测试...")
        error_test_cases = [
            "",  # 空字符串
            "a" * 1000,  # 超长字符串
            "🙂😊😢😡",  # 特殊字符
            None  # None值
        ]
        
        error_handling_results = []
        for test_input in error_test_cases:
            try:
                if test_input is not None:
                    single_predictions = self.basic_fusion.get_single_predictions(test_input)
                    result = self.basic_fusion.fusion_strategies['simple_voting'](single_predictions)
                    error_handling_results.append({'input': str(test_input)[:50], 'status': 'success'})
                else:
                    error_handling_results.append({'input': 'None', 'status': 'skipped'})
            except Exception as e:
                error_handling_results.append({'input': str(test_input)[:50], 'status': f'error: {str(e)[:100]}'})
        
        stability_results['error_handling'] = error_handling_results
        
        for result in error_handling_results:
            print(f"  输入: {result['input'][:30]}... -> {result['status']}")
        
        self.test_results['stability'] = stability_results
        return stability_results
    
    def generate_test_report(self):
        """生成测试报告"""
        print("\n📋 生成测试报告...")
        print("="*80)
        
        report = {
            'test_time': datetime.now().isoformat(),
            'summary': {},
            'detailed_results': self.test_results
        }
        
        # 基础融合策略总结
        if 'basic_fusion' in self.test_results:
            basic_summary = {}
            for strategy, results in self.test_results['basic_fusion'].items():
                if results['total'] > 0:
                    accuracy = results['correct'] / results['total']
                    avg_time = np.mean(results['execution_times']) if results['execution_times'] else 0
                    basic_summary[strategy] = {
                        'accuracy': accuracy,
                        'avg_execution_time': avg_time,
                        'error_count': len(results['errors'])
                    }
            report['summary']['basic_fusion'] = basic_summary
        
        # 高级融合策略总结
        if 'advanced_fusion' in self.test_results:
            advanced_summary = {}
            for strategy, results in self.test_results['advanced_fusion'].items():
                avg_time = np.mean(results['execution_times']) if results['execution_times'] else 0
                advanced_summary[strategy] = {
                    'prediction_count': len(results['predictions']),
                    'avg_execution_time': avg_time,
                    'error_count': len(results['errors'])
                }
            report['summary']['advanced_fusion'] = advanced_summary
        
        # 稳定性测试总结
        if 'stability' in self.test_results:
            stability_summary = {}
            if 'repeated_predictions' in self.test_results['stability']:
                stability_summary['consistency_rate'] = self.test_results['stability']['repeated_predictions'].get('consistency_rate', 0)
            report['summary']['stability'] = stability_summary
        
        # 保存报告
        report_file = f"d:\\bigcreate\\test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ 测试报告已保存: {report_file}")
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
        
        # 显示总结
        print(f"\n📊 测试总结:")
        print(f"  测试时间: {report['test_time']}")
        
        if 'basic_fusion' in report['summary']:
            print(f"\n  基础融合策略性能:")
            for strategy, metrics in report['summary']['basic_fusion'].items():
                print(f"    {strategy}: 准确率={metrics['accuracy']:.3f}, 平均时间={metrics['avg_execution_time']:.4f}s")
        
        if 'advanced_fusion' in report['summary']:
            print(f"\n  高级融合策略性能:")
            for strategy, metrics in report['summary']['advanced_fusion'].items():
                print(f"    {strategy}: 预测数={metrics['prediction_count']}, 平均时间={metrics['avg_execution_time']:.4f}s")
        
        if 'stability' in report['summary']:
            consistency = report['summary']['stability'].get('consistency_rate', 0)
            print(f"\n  系统稳定性: 一致性率={consistency:.3f}")
        
        return report
    
    def run_full_test_suite(self):
        """运行完整测试套件"""
        print("🚀 开始多模态融合系统全面测试")
        print("="*80)
        
        if not self.initialize_systems():
            print("❌ 系统初始化失败，测试终止")
            return
        
        # 执行所有测试
        try:
            self.test_basic_fusion_strategies()
            self.test_advanced_fusion_strategies()
            self.test_system_stability()
            
            # 生成报告
            report = self.generate_test_report()
            
            print(f"\n🎉 测试完成！")
            return report
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            return None

def main():
    """主函数"""
    test_framework = SystemTestFramework()
    report = test_framework.run_full_test_suite()
    
    if report:
        print("\n✅ 测试成功完成，请查看生成的测试报告")
    else:
        print("\n❌ 测试未能完成")

if __name__ == "__main__":
    main()
