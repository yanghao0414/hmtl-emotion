#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL数据集预处理 V3 - 去重+清洗+按对话划分
1. 去掉完全重复的文本（同标签只保留一条）
2. 过滤过短/无实质内容的句子
3. 过滤模板化回复（"谢谢你的建议"等）
4. 按对话文件划分，防止数据泄露
"""

import json
import os
import re
import random
from collections import Counter, defaultdict

INPUT_DIR = r'd:\silent like onion\03_数据\文本已标注数据'
OUTPUT_DIR = r'd:\bigcreate\05_数据文件'
TRAIN_RATIO = 0.85
RANDOM_SEED = 42

EMOTION_7_MAP = {
    '愤怒': 0, '焦虑': 1, '快乐': 2, '悲伤': 3,
    '失望': 4, '支持': 5, '平静': 6,
    '生气': 0,
    '紧张': 1, '担心': 1, '害怕': 1, '恐惧': 1, '困惑': 1, '犹豫': 1,
    '兴奋': 2, '激动': 2, '希望': 2, '期待': 2, '自信': 2,
    '沮丧': 3, '无助': 3,
    '理解': 5, '安慰': 5, '鼓励': 5,
    '放松': 6, '坚定': 6,
    '感激': 2, '果断': 6,
}

LABEL_7_TO_4 = {0:1, 1:1, 2:0, 3:2, 4:2, 5:0, 6:3}
LABEL_4_TO_3 = {0:0, 1:1, 2:1, 3:2}

AROUSAL_MAP = {0:0.85, 1:0.75, 2:0.70, 3:0.30, 4:0.35, 5:0.50, 6:0.25}
VALENCE_MAP = {0:-0.80, 1:-0.60, 2:0.75, 3:-0.70, 4:-0.50, 5:0.60, 6:0.05}

# 模板化回复模式（正则）
TEMPLATE_PATTERNS = [
    r'^谢谢你?的?(建议|支持|鼓励|帮助|理解|关心|倾听|回复|安慰)[。！!和与]*',
    r'^再次感谢你?的?(建议|支持|鼓励|帮助|理解|关心)[。！!]*$',
    r'^谢谢你?[。！!]*$',
    r'^好的[，,]?我(会|知道|明白|了解)了?[。！!]*$',
    r'^嗯[，,]?我(明白|知道|了解)了?[。！!]*$',
    r'^我明白了[。！!]*$',
    r'^好的[。！!]*$',
    r'^嗯[。！!]*$',
    r'^是的[。！!]*$',
    r'^对[。！!]*$',
    r'^我知道了[。！!]*$',
    r'^我了解了[。！!]*$',
]

TEMPLATE_REGEXES = [re.compile(p) for p in TEMPLATE_PATTERNS]

MIN_TEXT_LENGTH = 6  # 至少6个字符


def is_template(text):
    """判断是否为模板化回复"""
    for regex in TEMPLATE_REGEXES:
        if regex.match(text.strip()):
            return True
    return False


def process_files():
    """读取所有对话文件，按文件分组"""
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    
    file_samples = {}
    stats = {'total': 0, 'short': 0, 'template': 0, 'kept': 0, 'no_label': 0}
    
    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            data = json.load(open(fpath, 'r', encoding='utf-8'))
        except Exception:
            continue
        
        if not isinstance(data, list):
            continue
        
        samples = []
        for msg in data:
            if msg.get('role') != 'client' or 'annotation' not in msg:
                continue
            for ann in msg['annotation']:
                sentence = ann.get('sentence', '').strip()
                label_name = ann.get('label', '').strip()
                stats['total'] += 1
                
                if not sentence or not label_name:
                    stats['no_label'] += 1
                    continue
                if label_name not in EMOTION_7_MAP:
                    stats['no_label'] += 1
                    continue
                
                # 过滤过短文本
                if len(sentence) < MIN_TEXT_LENGTH:
                    stats['short'] += 1
                    continue
                
                # 过滤模板化回复
                if is_template(sentence):
                    stats['template'] += 1
                    continue
                
                label_7 = EMOTION_7_MAP[label_name]
                label_4 = LABEL_7_TO_4[label_7]
                label_3 = LABEL_4_TO_3[label_4]
                
                noise_a = random.uniform(-0.08, 0.08)
                noise_v = random.uniform(-0.08, 0.08)
                arousal = max(0.0, min(1.0, AROUSAL_MAP[label_7] + noise_a))
                valence = max(-1.0, min(1.0, VALENCE_MAP[label_7] + noise_v))
                
                samples.append({
                    'text': sentence,
                    'original_emotion': label_name,
                    'label_7': label_7,
                    'label_4': label_4,
                    'label_3': label_3,
                    'arousal': round(arousal, 4),
                    'valence': round(valence, 4),
                    'source': fname,
                })
                stats['kept'] += 1
        
        if samples:
            file_samples[fname] = samples
    
    print(f"原始句子总数: {stats['total']}")
    print(f"  过短(<{MIN_TEXT_LENGTH}字): {stats['short']}")
    print(f"  模板化回复: {stats['template']}")
    print(f"  无标签/未知标签: {stats['no_label']}")
    print(f"  保留: {stats['kept']}")
    print(f"  对话文件数: {len(file_samples)}")
    
    return file_samples


def deduplicate_within_files(file_samples):
    """在每个文件内去重（同文本+同标签只保留一条）"""
    total_before = sum(len(v) for v in file_samples.values())
    
    for fname in file_samples:
        seen = set()
        unique = []
        for s in file_samples[fname]:
            key = (s['text'], s['label_4'])
            if key not in seen:
                seen.add(key)
                unique.append(s)
        file_samples[fname] = unique
    
    total_after = sum(len(v) for v in file_samples.values())
    print(f"\n文件内去重: {total_before} -> {total_after} (去掉{total_before-total_after}条)")
    return file_samples


def deduplicate_global(file_samples):
    """全局去重：同一文本+同一标签，只在第一个出现的文件中保留"""
    seen_global = {}  # (text, label) -> first_filename
    removed = 0
    
    # 按文件名排序确保确定性
    for fname in sorted(file_samples.keys()):
        unique = []
        for s in file_samples[fname]:
            key = (s['text'], s['label_4'])
            if key not in seen_global:
                seen_global[key] = fname
                unique.append(s)
            else:
                removed += 1
        file_samples[fname] = unique
    
    # 删除空文件
    file_samples = {k: v for k, v in file_samples.items() if v}
    
    total = sum(len(v) for v in file_samples.values())
    print(f"全局去重: 去掉{removed}条, 剩余{total}条, {len(file_samples)}个对话")
    return file_samples


def split_by_file(file_samples):
    """按对话文件划分"""
    random.seed(RANDOM_SEED)
    
    filenames = list(file_samples.keys())
    random.shuffle(filenames)
    
    split_idx = int(len(filenames) * TRAIN_RATIO)
    train_files = filenames[:split_idx]
    eval_files = filenames[split_idx:]
    
    train_data = []
    for fname in train_files:
        for i, s in enumerate(file_samples[fname]):
            s['id'] = f"{fname}_{i}"
            train_data.append(s)
    
    eval_data = []
    for fname in eval_files:
        for i, s in enumerate(file_samples[fname]):
            s['id'] = f"{fname}_{i}"
            eval_data.append(s)
    
    # 验证
    train_sources = set(s['source'] for s in train_data)
    eval_sources = set(s['source'] for s in eval_data)
    overlap = train_sources & eval_sources
    assert len(overlap) == 0, f"数据泄露！重叠: {len(overlap)}"
    
    train_texts = set(s['text'] for s in train_data)
    eval_texts = set(s['text'] for s in eval_data)
    text_overlap = train_texts & eval_texts
    
    # 保存
    train_path = os.path.join(OUTPUT_DIR, 'full_training_set_hmtl_v3.json')
    eval_path = os.path.join(OUTPUT_DIR, 'full_eval_set_hmtl_v3.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n训练集: {len(train_data)} 条 ({len(train_files)} 个对话)")
    print(f"评估集: {len(eval_data)} 条 ({len(eval_files)} 个对话)")
    print(f"来源文件重叠: {len(overlap)}")
    print(f"文本重叠: {len(text_overlap)} ({len(text_overlap)/max(len(eval_texts),1)*100:.1f}%)")
    
    label_4_names = {0:'积极', 1:'激活消极', 2:'非激活消极', 3:'平静'}
    for name, data in [('训练集', train_data), ('评估集', eval_data)]:
        c = Counter(s['label_4'] for s in data)
        print(f"\n{name} 4类分布:")
        for k in sorted(c):
            print(f"  {label_4_names[k]}: {c[k]} ({c[k]/len(data)*100:.1f}%)")
    
    # 高频文本检查
    all_texts = [s['text'] for s in train_data + eval_data]
    text_counts = Counter(all_texts)
    top_repeat = text_counts.most_common(5)
    print(f"\n清洗后最高频文本:")
    for text, count in top_repeat:
        print(f"  [{count}次] {text[:50]}")
    
    return train_path, eval_path


if __name__ == '__main__':
    print("=" * 60)
    print("HMTL 完整数据集预处理 V3（去重+清洗+按对话划分）")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    file_samples = process_files()
    file_samples = deduplicate_within_files(file_samples)
    file_samples = deduplicate_global(file_samples)
    train_path, eval_path = split_by_file(file_samples)
    
    print(f"\n✅ V3数据预处理完成！")
    print(f"训练集: {train_path}")
    print(f"评估集: {eval_path}")
