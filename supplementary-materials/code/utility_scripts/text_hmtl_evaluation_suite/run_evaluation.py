#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text HMTL 模型评估脚本
用于在新收集的独立测试集上评估模型真实性能

使用方法:
    python run_evaluation.py --data annotated_data.csv --model best_model_v2.pt
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score
)

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '02_模型代码'))

from transformers import BertTokenizer

# 情绪标签
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}
EMOTION_7_LABELS = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']


class TextHMTLEvaluator:
    """Text HMTL 模型评估器"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 模型文件路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载分词器
        print("加载分词器...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("模型加载完成!")
    
    def _load_model(self, model_path: str):
        """加载模型"""
        from hmtl_model_v2 import HMTLEmotionModelV2
        
        model = HMTLEmotionModelV2()
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        return model
    
    def predict(self, text: str) -> dict:
        """
        预测单条文本
        
        Returns:
            dict: {
                'label_7': int,
                'label_7_name': str,
                'label_4': int,
                'label_3': int,
                'arousal': float,
                'valence': float,
                'probs_7': list  # 7类概率
            }
        """
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # 解析结果
        probs_7 = torch.softmax(outputs['label_7_logits'], dim=1)[0].cpu().numpy()
        label_7 = int(outputs['label_7_logits'].argmax(dim=1)[0].item())
        label_4 = int(outputs['label_4_logits'].argmax(dim=1)[0].item())
        label_3 = int(outputs['label_3_logits'].argmax(dim=1)[0].item())
        arousal = float(outputs['arousal'][0].item())
        valence = float(outputs['valence'][0].item())
        
        return {
            'label_7': label_7,
            'label_7_name': EMOTION_7_NAMES[label_7],
            'label_4': label_4,
            'label_3': label_3,
            'arousal': arousal,
            'valence': valence,
            'probs_7': probs_7.tolist()
        }
    
    def predict_batch(self, texts: list) -> list:
        """批量预测"""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def load_annotated_data(csv_path: str) -> pd.DataFrame:
    """加载标注数据"""
    df = pd.read_csv(csv_path)
    
    # 验证必要列
    required_cols = ['id', 'text', 'label_7', 'arousal', 'valence']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    
    # 过滤无效数据
    df = df.dropna(subset=['text', 'label_7'])
    df['label_7'] = df['label_7'].astype(int)
    
    print(f"加载数据: {len(df)} 条")
    return df


def evaluate_classification(y_true: list, y_pred: list, labels: list = None) -> dict:
    """评估分类性能"""
    labels = labels or list(range(7))
    
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
    
    # 每类F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # 分类报告
    report = classification_report(y_true, y_pred, labels=labels, 
                                   target_names=EMOTION_7_LABELS, 
                                   zero_division=0, output_dict=True)
    
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': {EMOTION_7_LABELS[i]: f for i, f in enumerate(per_class_f1)},
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def evaluate_regression(y_true: list, y_pred: list) -> dict:
    """评估回归性能"""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    
    return {
        'mae': mae,
        'r2': r2
    }


def analyze_by_source(df: pd.DataFrame, predictions: list) -> dict:
    """按数据来源分析"""
    if 'source' not in df.columns:
        return {}
    
    results = {}
    for source in df['source'].unique():
        mask = df['source'] == source
        y_true = df.loc[mask, 'label_7'].tolist()
        y_pred = [predictions[i]['label_7'] for i in df[mask].index]
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            results[source] = {
                'count': len(y_true),
                'accuracy': acc
            }
    
    return results


def generate_report(eval_results: dict, output_path: str):
    """生成评估报告"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Text HMTL 模型真实性能评估报告")
    report_lines.append("=" * 70)
    report_lines.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"评估样本数: {eval_results['total_samples']}")
    report_lines.append("")
    
    # 7类分类性能
    report_lines.append("-" * 70)
    report_lines.append("【7类情绪分类性能】")
    report_lines.append("-" * 70)
    cls_7 = eval_results['classification_7']
    report_lines.append(f"准确率 (Accuracy): {cls_7['accuracy']*100:.2f}%")
    report_lines.append(f"宏平均 F1 (Macro-F1): {cls_7['macro_f1']*100:.2f}%")
    report_lines.append(f"加权 F1 (Weighted-F1): {cls_7['weighted_f1']*100:.2f}%")
    report_lines.append("")
    
    report_lines.append("每类 F1 分数:")
    for emotion, f1 in cls_7['per_class_f1'].items():
        report_lines.append(f"  {emotion}: {f1*100:.2f}%")
    report_lines.append("")
    
    # 混淆矩阵
    report_lines.append("混淆矩阵:")
    report_lines.append("        " + "  ".join([f"{e[:2]:>4}" for e in EMOTION_7_LABELS]))
    cm = cls_7['confusion_matrix']
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:4d}" for v in row])
        report_lines.append(f"{EMOTION_7_LABELS[i][:2]:>6}  {row_str}")
    report_lines.append("")
    
    # 回归性能
    report_lines.append("-" * 70)
    report_lines.append("【Arousal/Valence 回归性能】")
    report_lines.append("-" * 70)
    reg_a = eval_results['regression_arousal']
    reg_v = eval_results['regression_valence']
    report_lines.append(f"Arousal MAE: {reg_a['mae']:.4f}")
    report_lines.append(f"Arousal R²: {reg_a['r2']:.4f}")
    report_lines.append(f"Valence MAE: {reg_v['mae']:.4f}")
    report_lines.append(f"Valence R²: {reg_v['r2']:.4f}")
    report_lines.append("")
    
    # 按来源分析
    if eval_results.get('by_source'):
        report_lines.append("-" * 70)
        report_lines.append("【按数据来源分析】")
        report_lines.append("-" * 70)
        for source, stats in eval_results['by_source'].items():
            report_lines.append(f"{source}: {stats['count']}条, 准确率={stats['accuracy']*100:.2f}%")
        report_lines.append("")
    
    # 结论
    report_lines.append("-" * 70)
    report_lines.append("【评估结论】")
    report_lines.append("-" * 70)
    acc = cls_7['accuracy']
    if acc >= 0.85:
        conclusion = "✅ 优秀: 模型泛化能力强，可直接用于生产"
    elif acc >= 0.75:
        conclusion = "✅ 良好: 模型性能可接受，建议针对性优化"
    elif acc >= 0.65:
        conclusion = "⚠️ 一般: 需要扩充训练数据，重点改进弱势类别"
    else:
        conclusion = "❌ 较差: 模型过拟合严重，需要重新设计训练策略"
    report_lines.append(conclusion)
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到: {output_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Text HMTL 模型评估')
    parser.add_argument('--data', type=str, required=True, help='标注数据CSV路径')
    parser.add_argument('--model', type=str, 
                        default=r'd:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt',
                        help='模型文件路径')
    parser.add_argument('--output', type=str, default='evaluation_report.txt',
                        help='输出报告路径')
    args = parser.parse_args()
    
    # 加载数据
    print("\n" + "="*60)
    print("加载标注数据...")
    df = load_annotated_data(args.data)
    
    # 显示数据分布
    print("\n7类情绪分布:")
    for label_id in sorted(df['label_7'].unique()):
        count = len(df[df['label_7'] == label_id])
        name = EMOTION_7_NAMES.get(label_id, '未知')
        print(f"  [{label_id}] {name}: {count} ({count/len(df)*100:.1f}%)")
    
    # 初始化评估器
    print("\n" + "="*60)
    evaluator = TextHMTLEvaluator(args.model)
    
    # 预测
    print("\n" + "="*60)
    print("开始预测...")
    texts = df['text'].tolist()
    predictions = evaluator.predict_batch(texts)
    print(f"预测完成: {len(predictions)} 条")
    
    # 评估
    print("\n" + "="*60)
    print("计算评估指标...")
    
    y_true_7 = df['label_7'].tolist()
    y_pred_7 = [p['label_7'] for p in predictions]
    
    y_true_arousal = df['arousal'].tolist()
    y_pred_arousal = [p['arousal'] for p in predictions]
    
    y_true_valence = df['valence'].tolist()
    y_pred_valence = [p['valence'] for p in predictions]
    
    eval_results = {
        'total_samples': len(df),
        'classification_7': evaluate_classification(y_true_7, y_pred_7),
        'regression_arousal': evaluate_regression(y_true_arousal, y_pred_arousal),
        'regression_valence': evaluate_regression(y_true_valence, y_pred_valence),
        'by_source': analyze_by_source(df, predictions)
    }
    
    # 保存详细结果
    results_json_path = args.output.replace('.txt', '_detailed.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {results_json_path}")
    
    # 生成报告
    print("\n" + "="*60)
    generate_report(eval_results, args.output)


if __name__ == '__main__':
    main()
