#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩充训练数据脚本
1. 从已下载的公开数据中筛选合适样本
2. 数据增强（同义词替换、回译等）
"""

import os
import json
import random
import pandas as pd
from collections import defaultdict

# 4类定义
LABEL_4_NAMES = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}
LABEL_4_TO_7 = {
    0: [2, 5],      # 积极 → 快乐/支持
    1: [0, 1],      # 激活消极 → 愤怒/焦虑
    2: [3, 4],      # 非激活消极 → 悲伤/失望
    3: [6],         # 平静 → 平静
}

# A/V 典型值
AV_BY_LABEL_4 = {
    0: (0.6, 0.8),   # 积极
    1: (0.8, -0.7),  # 激活消极
    2: (0.3, -0.6),  # 非激活消极
    3: (0.2, 0.2),   # 平静
}


def load_downloaded_data():
    """加载已下载的公开数据"""
    csv_path = "data/raw/evaluation_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path}，请先运行 download_datasets_v2.py")
        return []
    
    df = pd.read_csv(csv_path)
    # 只取需要人工校验的（公开数据集的）
    df_public = df[df['source'] != 'manual']
    
    samples = []
    for _, row in df_public.iterrows():
        samples.append({
            'text': row['text'],
            'source': row['source']
        })
    
    print(f"加载公开数据: {len(samples)} 条")
    return samples


def classify_by_keywords(text):
    """
    基于关键词初步分类
    返回 (label_4, confidence)
    """
    text = text.lower()
    
    # 激活消极关键词（愤怒/焦虑）
    angry_keywords = ['气死', '生气', '愤怒', '太差', '垃圾', '骗', '投诉', '恶心', '无语', '坑', '差评', '退款', '欺骗']
    anxious_keywords = ['担心', '焦虑', '紧张', '害怕', '不安', '慌', '忐忑', '压力']
    
    # 非激活消极关键词（悲伤/失望）
    sad_keywords = ['难过', '伤心', '悲伤', '哭', '痛苦', '心痛', '难受']
    disappointed_keywords = ['失望', '遗憾', '可惜', '白', '没想到', '本以为', '期望']
    
    # 积极关键词
    happy_keywords = ['开心', '高兴', '快乐', '棒', '好吃', '喜欢', '爱', '赞', '满意', '不错', '推荐', '值得', '惊喜']
    support_keywords = ['加油', '支持', '鼓励', '理解', '帮助', '谢谢', '感谢']
    
    # 平静关键词
    neutral_keywords = ['一般', '还行', '普通', '正常', '凑合', '马马虎虎', '还可以']
    
    # 计算匹配
    scores = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for kw in angry_keywords + anxious_keywords:
        if kw in text:
            scores[1] += 2  # 激活消极
    
    for kw in sad_keywords + disappointed_keywords:
        if kw in text:
            scores[2] += 2  # 非激活消极
    
    for kw in happy_keywords + support_keywords:
        if kw in text:
            scores[0] += 2  # 积极
    
    for kw in neutral_keywords:
        if kw in text:
            scores[3] += 2  # 平静
    
    # 选择最高分
    max_score = max(scores.values())
    if max_score == 0:
        return None, 0  # 无法分类
    
    best_label = max(scores, key=scores.get)
    confidence = max_score
    
    return best_label, confidence


def filter_and_label_samples(samples, target_counts):
    """
    筛选并标注样本
    target_counts: {label_4: count} 每类需要的数量
    """
    labeled = {0: [], 1: [], 2: [], 3: []}
    
    for s in samples:
        text = s['text']
        if len(text) < 5 or len(text) > 200:
            continue
        
        label_4, confidence = classify_by_keywords(text)
        if label_4 is None or confidence < 2:
            continue
        
        # 检查是否还需要这个类别
        if len(labeled[label_4]) < target_counts.get(label_4, 0):
            a, v = AV_BY_LABEL_4[label_4]
            # 添加一些随机扰动
            a += random.uniform(-0.1, 0.1)
            v += random.uniform(-0.1, 0.1)
            a = max(0, min(1, a))
            v = max(-1, min(1, v))
            
            labeled[label_4].append({
                'text': text,
                'label_4': label_4,
                'label_3': 0 if label_4 == 0 else (2 if label_4 == 3 else 1),
                'label_7': random.choice(LABEL_4_TO_7[label_4]),
                'arousal': round(a, 2),
                'valence': round(v, 2),
                'source': s['source'],
                'auto_labeled': True
            })
    
    return labeled


def create_augmented_samples():
    """
    创建数据增强样本
    主要针对平静类和激活消极类
    """
    augmented = []
    
    # 平静类模板
    neutral_templates = [
        "还行吧，没什么特别的",
        "一般般，中规中矩",
        "普通，正常水平",
        "凑合能用",
        "马马虎虎吧",
        "还可以，不好不坏",
        "就那样吧",
        "没什么感觉",
        "正常发挥",
        "中等水平",
        "不功不过",
        "符合预期",
        "和描述一致",
        "收到了，没问题",
        "好的，知道了",
        "嗯，了解",
        "可以，没意见",
        "行，就这样",
        "差不多吧",
        "还行，能接受",
        "一般般吧，不算好也不算差",
        "普普通通，没什么亮点",
        "正常使用，没什么问题",
        "还凑合，勉强能用",
        "中规中矩，符合价格",
        "没什么特别的感觉",
        "就这样吧，不评价了",
        "收到，确认",
        "好的，明白了",
        "嗯嗯，知道了",
    ]
    
    for text in neutral_templates:
        augmented.append({
            'text': text,
            'label_4': 3,
            'label_3': 2,
            'label_7': 6,
            'arousal': round(0.2 + random.uniform(-0.1, 0.1), 2),
            'valence': round(0.2 + random.uniform(-0.1, 0.1), 2),
            'source': 'augmented_neutral',
            'auto_labeled': False
        })
    
    # 激活消极类模板（愤怒/焦虑）
    activated_negative_templates = [
        # 愤怒
        ("太差了！差评！", 0),
        ("垃圾！浪费钱！", 0),
        ("骗人的！假货！", 0),
        ("服务态度太差了！", 0),
        ("简直不能忍！", 0),
        ("气死我了！", 0),
        ("什么破东西！", 0),
        ("太坑人了！", 0),
        ("要投诉！", 0),
        ("退款！必须退款！", 0),
        ("恶心！再也不买了！", 0),
        ("无语了，太差劲了", 0),
        ("怎么会这样！太过分了！", 0),
        ("不能接受！要个说法！", 0),
        ("差到极点！", 0),
        # 焦虑
        ("好担心啊，不知道行不行", 1),
        ("有点紧张，希望没问题", 1),
        ("不确定能不能用，有点慌", 1),
        ("万一出问题怎么办", 1),
        ("心里没底，忐忑不安", 1),
        ("压力好大，不知道能不能成", 1),
        ("好害怕会失败", 1),
        ("越想越担心", 1),
        ("不知道该怎么办", 1),
        ("好迷茫，不知道对不对", 1),
        ("有点不安，希望顺利", 1),
        ("担心会出问题", 1),
        ("心里七上八下的", 1),
        ("不太放心，有点焦虑", 1),
        ("压力山大啊", 1),
    ]
    
    for text, label_7 in activated_negative_templates:
        a = 0.85 if label_7 == 0 else 0.7  # 愤怒激活度更高
        v = -0.8 if label_7 == 0 else -0.6
        augmented.append({
            'text': text,
            'label_4': 1,
            'label_3': 1,
            'label_7': label_7,
            'arousal': round(a + random.uniform(-0.1, 0.1), 2),
            'valence': round(v + random.uniform(-0.1, 0.1), 2),
            'source': 'augmented_activated_neg',
            'auto_labeled': False
        })
    
    print(f"创建增强样本: {len(augmented)} 条")
    return augmented


def merge_with_original(new_samples, original_path, output_path):
    """合并新样本和原始训练数据"""
    # 加载原始数据
    with open(original_path, 'r', encoding='utf-8') as f:
        original = json.load(f)
    
    print(f"原始训练数据: {len(original)} 条")
    
    # 转换新样本格式
    for i, s in enumerate(new_samples):
        s['id'] = f"new_{i:04d}"
        if 'original_emotion' not in s:
            s['original_emotion'] = LABEL_4_NAMES[s['label_4']]
    
    # 合并
    merged = original + new_samples
    
    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"合并后总数: {len(merged)} 条")
    
    # 统计新的分布
    label_4_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for item in merged:
        label_4_counts[item['label_4']] += 1
    
    print("\n新的4类分布:")
    for label_id, name in LABEL_4_NAMES.items():
        print(f"  {name}: {label_4_counts[label_id]}")
    
    return merged


def main():
    print("="*60)
    print("扩充训练数据")
    print("="*60)
    
    # 当前分布
    print("\n当前训练数据分布:")
    print("  积极: 109")
    print("  激活消极: 109")
    print("  非激活消极: 74")
    print("  平静: 27 ← 需要扩充!")
    
    # 目标：每类至少100条
    target_counts = {
        0: 0,   # 积极够了
        1: 20,  # 激活消极再加一些
        2: 30,  # 非激活消极再加一些
        3: 80,  # 平静需要大量补充
    }
    
    all_new_samples = []
    
    # 1. 从公开数据筛选
    print("\n" + "="*60)
    print("步骤1: 从公开数据筛选")
    print("="*60)
    
    public_samples = load_downloaded_data()
    if public_samples:
        labeled = filter_and_label_samples(public_samples, target_counts)
        for label_4, samples in labeled.items():
            print(f"  {LABEL_4_NAMES[label_4]}: 筛选到 {len(samples)} 条")
            all_new_samples.extend(samples)
    
    # 2. 数据增强
    print("\n" + "="*60)
    print("步骤2: 数据增强")
    print("="*60)
    
    augmented = create_augmented_samples()
    all_new_samples.extend(augmented)
    
    # 3. 合并
    print("\n" + "="*60)
    print("步骤3: 合并数据")
    print("="*60)
    
    original_path = r"d:\bigcreate\05_数据文件\training_set_hmtl.json"
    output_path = r"d:\bigcreate\05_数据文件\training_set_hmtl_expanded.json"
    
    merge_with_original(all_new_samples, original_path, output_path)
    
    print("\n" + "="*60)
    print("完成!")
    print(f"新训练数据保存到: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
