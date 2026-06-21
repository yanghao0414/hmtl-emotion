#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注数据验证脚本
检查标注数据的完整性和一致性

使用方法:
    python validate_annotations.py --data annotated_data.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# 情绪标签
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}
EMOTION_7_LABELS = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']

# 4类和3类映射
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}  # 7类→4类
LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}  # 7类→3类

# A/V 典型值
AV_TYPICAL = {
    0: (0.9, -0.8),   # 愤怒
    1: (0.7, -0.6),   # 焦虑
    2: (0.7, 0.9),    # 快乐
    3: (0.3, -0.7),   # 悲伤
    4: (0.4, -0.5),   # 失望
    5: (0.5, 0.7),    # 支持
    6: (0.2, 0.3),    # 平静
}


def validate_data(csv_path: str) -> dict:
    """验证标注数据"""
    print(f"加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    issues = []
    warnings = []
    stats = {}
    
    # 1. 检查必要列
    required_cols = ['id', 'text', 'label_7', 'arousal', 'valence']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(f"缺少必要列: {missing_cols}")
    
    # 2. 检查空值
    for col in required_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                issues.append(f"列 '{col}' 有 {null_count} 个空值")
    
    # 3. 检查 label_7 范围
    if 'label_7' in df.columns:
        invalid_labels = df[~df['label_7'].isin(range(7))]
        if len(invalid_labels) > 0:
            issues.append(f"label_7 有 {len(invalid_labels)} 个无效值 (应为0-6)")
    
    # 4. 检查 arousal 范围 (0-1)
    if 'arousal' in df.columns:
        invalid_arousal = df[(df['arousal'] < 0) | (df['arousal'] > 1)]
        if len(invalid_arousal) > 0:
            issues.append(f"arousal 有 {len(invalid_arousal)} 个超出范围 (应为0-1)")
    
    # 5. 检查 valence 范围 (-1 to 1)
    if 'valence' in df.columns:
        invalid_valence = df[(df['valence'] < -1) | (df['valence'] > 1)]
        if len(invalid_valence) > 0:
            issues.append(f"valence 有 {len(invalid_valence)} 个超出范围 (应为-1到1)")
    
    # 6. 检查 label_4 一致性 (如果存在)
    if 'label_4' in df.columns and 'label_7' in df.columns:
        df['expected_label_4'] = df['label_7'].map(LABEL_7_TO_4)
        inconsistent_4 = df[df['label_4'] != df['expected_label_4']]
        if len(inconsistent_4) > 0:
            warnings.append(f"label_4 有 {len(inconsistent_4)} 个与 label_7 不一致")
    
    # 7. 检查 label_3 一致性 (如果存在)
    if 'label_3' in df.columns and 'label_7' in df.columns:
        df['expected_label_3'] = df['label_7'].map(LABEL_7_TO_3)
        inconsistent_3 = df[df['label_3'] != df['expected_label_3']]
        if len(inconsistent_3) > 0:
            warnings.append(f"label_3 有 {len(inconsistent_3)} 个与 label_7 不一致")
    
    # 8. 检查 A/V 与情绪的一致性
    if all(c in df.columns for c in ['label_7', 'arousal', 'valence']):
        av_issues = []
        for idx, row in df.iterrows():
            label = int(row['label_7'])
            a, v = row['arousal'], row['valence']
            typical_a, typical_v = AV_TYPICAL.get(label, (0.5, 0))
            
            # 检查 arousal 偏差
            if abs(a - typical_a) > 0.4:
                av_issues.append(f"ID={row.get('id', idx)}: {EMOTION_7_NAMES[label]} 的 arousal={a:.2f} 偏离典型值 {typical_a}")
            
            # 检查 valence 符号
            if typical_v > 0 and v < -0.3:
                av_issues.append(f"ID={row.get('id', idx)}: {EMOTION_7_NAMES[label]} 应为正效价，但 valence={v:.2f}")
            elif typical_v < 0 and v > 0.3:
                av_issues.append(f"ID={row.get('id', idx)}: {EMOTION_7_NAMES[label]} 应为负效价，但 valence={v:.2f}")
        
        if av_issues:
            warnings.append(f"A/V 值与情绪不一致: {len(av_issues)} 处")
            for issue in av_issues[:5]:  # 只显示前5个
                warnings.append(f"  - {issue}")
            if len(av_issues) > 5:
                warnings.append(f"  ... 还有 {len(av_issues)-5} 处")
    
    # 9. 统计信息
    stats['total_samples'] = len(df)
    stats['valid_samples'] = len(df.dropna(subset=['text', 'label_7']))
    
    if 'label_7' in df.columns:
        stats['label_distribution'] = {}
        for label_id in range(7):
            count = len(df[df['label_7'] == label_id])
            stats['label_distribution'][EMOTION_7_NAMES[label_id]] = count
    
    if 'source' in df.columns:
        stats['source_distribution'] = df['source'].value_counts().to_dict()
    
    # 10. 文本长度统计
    if 'text' in df.columns:
        df['text_len'] = df['text'].astype(str).apply(len)
        stats['text_length'] = {
            'min': int(df['text_len'].min()),
            'max': int(df['text_len'].max()),
            'mean': float(df['text_len'].mean()),
            'median': float(df['text_len'].median())
        }
    
    return {
        'issues': issues,
        'warnings': warnings,
        'stats': stats
    }


def auto_fix(csv_path: str, output_path: str):
    """自动修复常见问题"""
    df = pd.read_csv(csv_path)
    fixed_count = 0
    
    # 1. 根据 label_7 自动填充 label_4 和 label_3
    if 'label_7' in df.columns:
        if 'label_4' not in df.columns or df['label_4'].isnull().any():
            df['label_4'] = df['label_7'].map(LABEL_7_TO_4)
            fixed_count += 1
            print("✓ 自动填充 label_4")
        
        if 'label_3' not in df.columns or df['label_3'].isnull().any():
            df['label_3'] = df['label_7'].map(LABEL_7_TO_3)
            fixed_count += 1
            print("✓ 自动填充 label_3")
    
    # 2. 根据 label_7 自动填充 label_7_name
    if 'label_7' in df.columns and 'label_7_name' not in df.columns:
        df['label_7_name'] = df['label_7'].map(EMOTION_7_NAMES)
        fixed_count += 1
        print("✓ 自动填充 label_7_name")
    
    # 3. 裁剪 arousal 到 [0, 1]
    if 'arousal' in df.columns:
        before = len(df[(df['arousal'] < 0) | (df['arousal'] > 1)])
        df['arousal'] = df['arousal'].clip(0, 1)
        if before > 0:
            fixed_count += 1
            print(f"✓ 裁剪 {before} 个 arousal 值到 [0, 1]")
    
    # 4. 裁剪 valence 到 [-1, 1]
    if 'valence' in df.columns:
        before = len(df[(df['valence'] < -1) | (df['valence'] > 1)])
        df['valence'] = df['valence'].clip(-1, 1)
        if before > 0:
            fixed_count += 1
            print(f"✓ 裁剪 {before} 个 valence 值到 [-1, 1]")
    
    # 保存
    df.to_csv(output_path, index=False)
    print(f"\n已保存修复后的数据到: {output_path}")
    print(f"共修复 {fixed_count} 类问题")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='验证标注数据')
    parser.add_argument('--data', type=str, required=True, help='标注数据CSV路径')
    parser.add_argument('--fix', action='store_true', help='自动修复常见问题')
    parser.add_argument('--output', type=str, help='修复后输出路径')
    args = parser.parse_args()
    
    # 验证
    print("="*60)
    print("标注数据验证")
    print("="*60)
    
    result = validate_data(args.data)
    
    # 显示统计
    print("\n【统计信息】")
    print(f"总样本数: {result['stats']['total_samples']}")
    print(f"有效样本数: {result['stats']['valid_samples']}")
    
    if 'label_distribution' in result['stats']:
        print("\n7类情绪分布:")
        for emotion, count in result['stats']['label_distribution'].items():
            pct = count / result['stats']['total_samples'] * 100
            print(f"  {emotion}: {count} ({pct:.1f}%)")
    
    if 'source_distribution' in result['stats']:
        print("\n数据来源分布:")
        for source, count in result['stats']['source_distribution'].items():
            print(f"  {source}: {count}")
    
    if 'text_length' in result['stats']:
        tl = result['stats']['text_length']
        print(f"\n文本长度: 最短={tl['min']}, 最长={tl['max']}, 平均={tl['mean']:.1f}")
    
    # 显示问题
    if result['issues']:
        print("\n【❌ 错误 - 必须修复】")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    if result['warnings']:
        print("\n【⚠️ 警告 - 建议检查】")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if not result['issues'] and not result['warnings']:
        print("\n✅ 数据验证通过，无问题!")
    
    # 自动修复
    if args.fix:
        print("\n" + "="*60)
        print("自动修复")
        print("="*60)
        output_path = args.output or args.data.replace('.csv', '_fixed.csv')
        auto_fix(args.data, output_path)
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
