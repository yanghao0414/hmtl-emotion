#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集辅助脚本
提供多种数据来源的收集模板和工具

使用方法:
    python collect_data.py --source weibo --output data/raw/weibo_raw.csv
"""

import os
import csv
import json
import argparse
from datetime import datetime


# 情绪标签参考
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}

# 典型 A/V 值
AV_TYPICAL = {
    0: (0.9, -0.8),   # 愤怒: 高激活，负效价
    1: (0.7, -0.6),   # 焦虑: 中高激活，负效价
    2: (0.7, 0.9),    # 快乐: 中高激活，正效价
    3: (0.3, -0.7),   # 悲伤: 低激活，负效价
    4: (0.4, -0.5),   # 失望: 低激活，负效价
    5: (0.5, 0.7),    # 支持: 中激活，正效价
    6: (0.2, 0.3),    # 平静: 低激活，中性
}

# 7类→4类映射
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}
# 7类→3类映射
LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}


def create_empty_template(output_path: str, source: str, count: int = 100):
    """创建空白标注模板"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'id', 'text', 'source', 'label_7', 'label_7_name',
            'label_4', 'label_3', 'arousal', 'valence',
            'annotator', 'confidence', 'notes'
        ])
        
        # 写入空行模板
        for i in range(count):
            writer.writerow([
                f'{source}_{i:04d}',  # id
                '',                    # text (待填写)
                source,                # source
                '',                    # label_7 (待填写)
                '',                    # label_7_name
                '',                    # label_4
                '',                    # label_3
                '',                    # arousal (待填写)
                '',                    # valence (待填写)
                '',                    # annotator
                '',                    # confidence
                ''                     # notes
            ])
    
    print(f"已创建空白模板: {output_path}")
    print(f"包含 {count} 行待填写数据")


def create_sample_data(output_path: str):
    """创建示例数据（用于测试）"""
    samples = [
        # 微博样本
        ("weibo_001", "今天被老板骂了一顿，真是太气人了！", "weibo", 0, 0.9, -0.8),
        ("weibo_002", "明天要考试了，好紧张啊，睡不着", "weibo", 1, 0.7, -0.6),
        ("weibo_003", "终于放假啦！开心到飞起~", "weibo", 2, 0.8, 0.9),
        ("weibo_004", "分手了，心里好难受", "weibo", 3, 0.3, -0.7),
        ("weibo_005", "等了一个月的快递，结果是坏的，太失望了", "weibo", 4, 0.4, -0.5),
        
        # 知乎样本
        ("zhihu_001", "我理解你的感受，这种情况确实很难处理", "zhihu", 5, 0.5, 0.7),
        ("zhihu_002", "这个问题其实很简单，按照以下步骤操作即可", "zhihu", 6, 0.2, 0.3),
        ("zhihu_003", "为什么总是这样？我真的很担心自己的未来", "zhihu", 1, 0.7, -0.6),
        
        # 客服样本
        ("customer_001", "你们这个服务太差了！我要投诉！", "customer", 0, 0.9, -0.8),
        ("customer_002", "谢谢你的帮助，问题解决了", "customer", 2, 0.6, 0.8),
        ("customer_003", "我的订单怎么还没发货？有点担心", "customer", 1, 0.6, -0.5),
        
        # 电商样本
        ("ecommerce_001", "质量很好，下次还会买", "ecommerce", 2, 0.6, 0.8),
        ("ecommerce_002", "和图片差太多了，很失望", "ecommerce", 4, 0.4, -0.5),
        ("ecommerce_003", "一般般吧，没什么特别的", "ecommerce", 6, 0.2, 0.1),
    ]
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'text', 'source', 'label_7', 'label_7_name',
            'label_4', 'label_3', 'arousal', 'valence',
            'annotator', 'confidence', 'notes'
        ])
        
        for sample in samples:
            id_, text, source, label_7, arousal, valence = sample
            writer.writerow([
                id_,
                text,
                source,
                label_7,
                EMOTION_7_NAMES[label_7],
                LABEL_7_TO_4[label_7],
                LABEL_7_TO_3[label_7],
                arousal,
                valence,
                'system',
                'high',
                '示例数据'
            ])
    
    print(f"已创建示例数据: {output_path}")
    print(f"包含 {len(samples)} 条样本")


def merge_data_files(input_files: list, output_path: str):
    """合并多个数据文件"""
    all_data = []
    
    for file_path in input_files:
        if os.path.exists(file_path):
            import pandas as pd
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"加载 {file_path}: {len(df)} 条")
    
    if all_data:
        import pandas as pd
        merged = pd.concat(all_data, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"\n合并完成: {output_path}")
        print(f"总计 {len(merged)} 条")
    else:
        print("没有找到有效的数据文件")


def print_annotation_guide():
    """打印标注指南"""
    print("""
================================================================================
                           情绪标注指南
================================================================================

【7类情绪定义】

  ID  情绪    定义                          典型表达
  --  ----    ----                          --------
  0   愤怒    高激活负面，带攻击性          "太气人了"、"简直无法忍受"
  1   焦虑    中高激活负面，带担忧          "好担心"、"不知道怎么办"
  2   快乐    高激活正面                    "太开心了"、"真棒"
  3   悲伤    低激活负面，带失落            "好难过"、"心里很难受"
  4   失望    低激活负面，期望落空          "真让人失望"、"没想到会这样"
  5   支持    中激活正面，带鼓励            "加油"、"我理解你"
  6   平静    低激活中性                    "还好吧"、"正常"

【Arousal/Valence 标注】

  Arousal (激活度): 0.0 ~ 1.0
    - 0.0-0.3: 低激活 (平静、悲伤)
    - 0.4-0.6: 中激活 (支持、失望)
    - 0.7-1.0: 高激活 (愤怒、焦虑、快乐)

  Valence (效价): -1.0 ~ 1.0
    - -1.0 ~ -0.3: 负面 (愤怒、焦虑、悲伤、失望)
    - -0.3 ~ 0.3: 中性 (平静)
    - 0.3 ~ 1.0: 正面 (快乐、支持)

【易混淆情绪区分】

  愤怒 vs 焦虑: 愤怒有攻击性，焦虑是担忧
  悲伤 vs 失望: 悲伤更深沉，失望有期望落空感
  支持 vs 快乐: 支持是给予他人的，快乐是自己的
  支持 vs 平静: 支持有情感投入，平静是中性的

【标注流程】

  1. 阅读文本，理解完整语境
  2. 判断主导情绪（7类之一）
  3. 评估 Arousal（激活程度）
  4. 评估 Valence（正负性）
  5. 标记置信度 (high/medium/low)
  6. 如有疑问，在 notes 中说明

================================================================================
""")


def main():
    parser = argparse.ArgumentParser(description='数据收集辅助工具')
    parser.add_argument('--action', type=str, default='template',
                        choices=['template', 'sample', 'merge', 'guide'],
                        help='操作类型')
    parser.add_argument('--source', type=str, default='weibo',
                        choices=['weibo', 'zhihu', 'customer', 'ecommerce'],
                        help='数据来源')
    parser.add_argument('--count', type=int, default=100, help='模板行数')
    parser.add_argument('--output', type=str, default='data/raw/template.csv',
                        help='输出路径')
    parser.add_argument('--inputs', type=str, nargs='+', help='合并时的输入文件列表')
    args = parser.parse_args()
    
    if args.action == 'template':
        create_empty_template(args.output, args.source, args.count)
    elif args.action == 'sample':
        create_sample_data(args.output)
    elif args.action == 'merge':
        if args.inputs:
            merge_data_files(args.inputs, args.output)
        else:
            print("请使用 --inputs 指定要合并的文件")
    elif args.action == 'guide':
        print_annotation_guide()


if __name__ == '__main__':
    main()
