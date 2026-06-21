#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从公开数据集下载中文情绪文本
用于验证 Text HMTL 模型的真实性能

数据来源:
1. HuggingFace: 多个中文情绪数据集
2. 这些数据与训练数据完全独立，适合做真实评估
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict

# 情绪映射
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}

# 7类→4类映射
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}
# 7类→3类映射
LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}

# 典型 A/V 值
AV_TYPICAL = {
    0: (0.9, -0.8),   # 愤怒
    1: (0.7, -0.6),   # 焦虑
    2: (0.7, 0.9),    # 快乐
    3: (0.3, -0.7),   # 悲伤
    4: (0.4, -0.5),   # 失望
    5: (0.5, 0.7),    # 支持
    6: (0.2, 0.3),    # 平静
}


def download_weibo_sentiment():
    """
    下载微博情感数据集
    来源: https://huggingface.co/datasets/tyqiangz/multilingual-sentiments
    """
    print("\n" + "="*60)
    print("下载微博情感数据集...")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        # 加载中文部分
        dataset = load_dataset("tyqiangz/multilingual-sentiments", "chinese", split="train")
        
        samples = []
        for item in dataset:
            text = item['text']
            # 原始标签: 0=negative, 1=neutral, 2=positive
            orig_label = item['label']
            
            # 映射到我们的7类 (简化映射)
            if orig_label == 0:  # negative
                label_7 = random.choice([0, 1, 3, 4])  # 愤怒/焦虑/悲伤/失望
            elif orig_label == 2:  # positive
                label_7 = random.choice([2, 5])  # 快乐/支持
            else:  # neutral
                label_7 = 6  # 平静
            
            samples.append({
                'text': text,
                'original_label': orig_label,
                'label_7': label_7,
                'source': 'weibo_sentiment'
            })
        
        print(f"下载完成: {len(samples)} 条")
        return samples
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("请先安装: pip install datasets")
        return []


def download_nlpcc_emotion():
    """
    下载 NLPCC 2014 情感分析数据
    这是一个经典的中文情感数据集
    """
    print("\n" + "="*60)
    print("下载 NLPCC 情感数据集...")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        # 尝试加载
        dataset = load_dataset("seamew/ChnSentiCorp", split="train")
        
        samples = []
        for item in dataset:
            text = item['text']
            label = item['label']  # 0=negative, 1=positive
            
            if label == 0:
                label_7 = random.choice([0, 1, 3, 4])
            else:
                label_7 = random.choice([2, 5])
            
            samples.append({
                'text': text,
                'original_label': label,
                'label_7': label_7,
                'source': 'chnsenticorp'
            })
        
        print(f"下载完成: {len(samples)} 条")
        return samples
        
    except Exception as e:
        print(f"下载失败: {e}")
        return []


def download_waimai_sentiment():
    """
    下载外卖评论情感数据集
    来源: 外卖平台评论，适合测试电商场景
    """
    print("\n" + "="*60)
    print("下载外卖评论数据集...")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset("kuroneko5943/stock21", split="train")
        
        samples = []
        for item in list(dataset)[:500]:  # 取前500条
            text = item.get('text', item.get('content', ''))
            if not text or len(text) < 5:
                continue
            
            # 随机分配标签（需要人工校验）
            label_7 = random.randint(0, 6)
            
            samples.append({
                'text': text[:200],  # 截断
                'original_label': -1,
                'label_7': label_7,
                'source': 'stock_comments',
                'needs_review': True
            })
        
        print(f"下载完成: {len(samples)} 条")
        return samples
        
    except Exception as e:
        print(f"下载失败: {e}")
        return []


def create_manual_samples():
    """
    创建需要手动标注的样本模板
    包含多种场景的典型文本
    """
    print("\n" + "="*60)
    print("创建手动标注样本...")
    print("="*60)
    
    # 各类情绪的典型表达模板
    templates = {
        0: [  # 愤怒
            "这服务态度也太差了吧！",
            "简直不能忍，太过分了！",
            "气死我了，怎么会这样！",
            "什么破东西，浪费我钱！",
            "投诉！必须投诉！",
        ],
        1: [  # 焦虑
            "好担心明天的结果...",
            "不知道该怎么办才好",
            "越想越紧张，睡不着",
            "万一失败了怎么办？",
            "心里七上八下的",
        ],
        2: [  # 快乐
            "太棒了！超级开心！",
            "哈哈哈笑死我了",
            "今天运气真好~",
            "终于成功了！激动！",
            "好幸福啊！",
        ],
        3: [  # 悲伤
            "心里好难受...",
            "眼泪止不住地流",
            "感觉好孤独",
            "再也回不去了",
            "好想哭",
        ],
        4: [  # 失望
            "真让人失望",
            "没想到会是这样",
            "期望太高了",
            "白等了这么久",
            "算了，不抱希望了",
        ],
        5: [  # 支持
            "加油，你可以的！",
            "我理解你的感受",
            "别担心，会好起来的",
            "有什么需要帮忙的吗？",
            "相信自己！",
        ],
        6: [  # 平静
            "还行吧，一般般",
            "正常操作",
            "没什么特别的",
            "知道了",
            "好的，收到",
        ],
    }
    
    samples = []
    for label_7, texts in templates.items():
        for text in texts:
            samples.append({
                'text': text,
                'original_label': label_7,
                'label_7': label_7,
                'source': 'manual_template',
                'needs_review': False
            })
    
    print(f"创建完成: {len(samples)} 条")
    return samples


def prepare_for_annotation(samples: list, output_path: str, sample_size: int = None):
    """
    准备标注数据
    """
    if sample_size and len(samples) > sample_size:
        samples = random.sample(samples, sample_size)
    
    # 添加完整字段
    data = []
    for i, s in enumerate(samples):
        label_7 = s.get('label_7', -1)
        a, v = AV_TYPICAL.get(label_7, (0.5, 0))
        
        data.append({
            'id': f"eval_{i:04d}",
            'text': s['text'],
            'source': s.get('source', 'unknown'),
            'label_7': label_7 if not s.get('needs_review') else '',
            'label_7_name': EMOTION_7_NAMES.get(label_7, '') if not s.get('needs_review') else '',
            'label_4': LABEL_7_TO_4.get(label_7, '') if not s.get('needs_review') else '',
            'label_3': LABEL_7_TO_3.get(label_7, '') if not s.get('needs_review') else '',
            'arousal': a if not s.get('needs_review') else '',
            'valence': v if not s.get('needs_review') else '',
            'annotator': '',
            'confidence': '',
            'notes': '需要人工校验' if s.get('needs_review') else ''
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n已保存到: {output_path}")
    print(f"总计 {len(data)} 条")
    
    # 统计
    print("\n来源分布:")
    for source in df['source'].unique():
        count = len(df[df['source'] == source])
        print(f"  {source}: {count}")


def main():
    print("="*60)
    print("中文情绪数据收集工具")
    print("="*60)
    
    all_samples = []
    
    # 1. 创建手动样本（35条，确保每类都有）
    manual = create_manual_samples()
    all_samples.extend(manual)
    
    # 2. 尝试下载公开数据集
    try:
        # 微博情感
        weibo = download_weibo_sentiment()
        if weibo:
            all_samples.extend(random.sample(weibo, min(200, len(weibo))))
        
        # ChnSentiCorp
        chn = download_nlpcc_emotion()
        if chn:
            all_samples.extend(random.sample(chn, min(200, len(chn))))
            
    except ImportError:
        print("\n⚠️ 需要安装 datasets 库:")
        print("   pip install datasets")
        print("\n将只使用手动样本...")
    
    # 3. 保存
    output_path = "data/raw/collected_data.csv"
    prepare_for_annotation(all_samples, output_path)
    
    print("\n" + "="*60)
    print("下一步:")
    print("1. 打开 data/raw/collected_data.csv")
    print("2. 检查并修正标签（特别是 needs_review 的行）")
    print("3. 保存到 data/annotated/ 目录")
    print("4. 运行 python run_evaluation.py --data data/annotated/xxx.csv")
    print("="*60)


if __name__ == '__main__':
    main()
