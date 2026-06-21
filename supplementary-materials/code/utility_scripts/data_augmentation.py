#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强脚本
解决"支持↔快乐"混淆问题

策略：
1. 同义词替换
2. 回译增强（中文→英文→中文）
3. 语义变换（保持情绪不变）
"""

import json
import random
from typing import List, Dict

# 针对"支持"类的语义模板
SUPPORT_TEMPLATES = [
    "{}，我支持你",
    "{}，我理解你的感受",
    "{}，我会帮助你的",
    "{}，你不是一个人",
    "{}，我陪着你",
    "{}，别担心",
    "{}，有我在",
    "{}，你能做到的",
    "{}，我相信你",
    "{}，加油"
]

# 针对"快乐"类的语义模板
HAPPY_TEMPLATES = [
    "{}，我很开心",
    "{}，太好了",
    "{}，真是太棒了",
    "{}，我很高兴",
    "{}，感觉真好",
    "{}，心情愉快",
    "{}，很满足",
    "{}，非常快乐",
    "{}，好开心啊",
    "{}，真幸福"
]

# 同义词词典
SYNONYMS = {
    '支持': ['理解', '帮助', '陪伴', '鼓励', '安慰'],
    '快乐': ['开心', '高兴', '愉快', '喜悦', '兴奋'],
    '好': ['棒', '不错', '很好', '优秀', '出色'],
    '谢谢': ['感谢', '多谢', '非常感谢'],
    '帮助': ['帮忙', '协助', '支持'],
    '理解': ['懂得', '明白', '体会']
}


def synonym_replacement(text: str, n=1) -> str:
    """
    同义词替换增强
    
    Args:
        text: 原始文本
        n: 替换词数
    
    Returns:
        增强后的文本
    """
    words = list(text)
    new_words = words.copy()
    
    for _ in range(n):
        for word, syns in SYNONYMS.items():
            if word in text:
                synonym = random.choice(syns)
                new_text = text.replace(word, synonym, 1)
                return new_text
    
    return text


def template_augmentation(text: str, emotion: str) -> List[str]:
    """
    模板增强
    
    Args:
        text: 原始文本
        emotion: 情绪类别（'支持' 或 '快乐'）
    
    Returns:
        增强样本列表
    """
    if emotion == '支持':
        templates = SUPPORT_TEMPLATES
    elif emotion == '快乐':
        templates = HAPPY_TEMPLATES
    else:
        return [text]
    
    # 提取核心内容（去除情绪词）
    core = text
    for keyword in ['支持', '理解', '帮助', '开心', '高兴', '快乐']:
        core = core.replace(keyword, '').strip()
    
    if not core:
        core = "这件事"
    
    # 生成增强样本
    augmented = []
    for template in templates[:3]:  # 每个样本生成3个增强版本
        try:
            new_text = template.format(core)
            if new_text != text:
                augmented.append(new_text)
        except:
            continue
    
    return augmented


def back_translation_simulation(text: str, emotion: str) -> str:
    """
    回译增强模拟
    （实际使用需要调用翻译API，这里用规则模拟）
    
    Args:
        text: 原始文本
        emotion: 情绪类别
    
    Returns:
        增强后的文本
    """
    # 简单的语序调整和同义词替换
    result = synonym_replacement(text, n=1)
    
    # 添加语气词
    if emotion == '支持' and not any(word in result for word in ['的', '了', '啊', '呢']):
        result = result.rstrip('。！？') + '啊'
    elif emotion == '快乐' and not result.endswith('！'):
        result = result.rstrip('。！？') + '！'
    
    return result


def augment_emotion_data(data: List[Dict], target_emotions: List[str], 
                         target_count: int = 80) -> List[Dict]:
    """
    为指定情绪类别增强数据
    
    Args:
        data: 原始数据列表
        target_emotions: 目标情绪列表（如['支持', '快乐']）
        target_count: 每个情绪的目标样本数
    
    Returns:
        增强后的数据列表
    """
    augmented_data = data.copy()
    
    for emotion in target_emotions:
        # 找到该情绪的所有样本
        emotion_samples = [item for item in data 
                          if item.get('original_emotion') == emotion]
        
        current_count = len(emotion_samples)
        print(f"\n处理 '{emotion}' 类:")
        print(f"  当前样本数: {current_count}")
        print(f"  目标样本数: {target_count}")
        
        if current_count >= target_count:
            print(f"  ✓ 已达到目标，无需增强")
            continue
        
        needed = target_count - current_count
        print(f"  需要增强: {needed} 条")
        
        # 增强策略
        generated = 0
        
        for sample in emotion_samples:
            if generated >= needed:
                break
            
            text = sample.get('text', '')
            
            # 策略1: 同义词替换
            if generated < needed:
                aug_text = synonym_replacement(text, n=1)
                if aug_text != text:
                    new_sample = sample.copy()
                    new_sample['text'] = aug_text
                    new_sample['id'] = f"aug_syn_{emotion}_{generated}"
                    new_sample['source'] = 'data_augmentation_synonym'
                    augmented_data.append(new_sample)
                    generated += 1
            
            # 策略2: 模板增强
            if generated < needed:
                aug_texts = template_augmentation(text, emotion)
                for aug_text in aug_texts[:min(2, needed - generated)]:
                    new_sample = sample.copy()
                    new_sample['text'] = aug_text
                    new_sample['id'] = f"aug_tmpl_{emotion}_{generated}"
                    new_sample['source'] = 'data_augmentation_template'
                    augmented_data.append(new_sample)
                    generated += 1
                    if generated >= needed:
                        break
            
            # 策略3: 回译模拟
            if generated < needed:
                aug_text = back_translation_simulation(text, emotion)
                if aug_text != text:
                    new_sample = sample.copy()
                    new_sample['text'] = aug_text
                    new_sample['id'] = f"aug_bt_{emotion}_{generated}"
                    new_sample['source'] = 'data_augmentation_backtranslation'
                    augmented_data.append(new_sample)
                    generated += 1
        
        print(f"  ✓ 实际生成: {generated} 条")
        print(f"  ✓ 总数: {current_count + generated}")
    
    return augmented_data


def main():
    """主函数"""
    print("="*60)
    print("数据增强脚本 - 解决支持↔快乐混淆")
    print("="*60)
    
    # 加载训练数据
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parents[1]
    train_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl.json")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"\n加载训练数据: {len(train_data)} 条")
    
    # 统计当前分布
    from collections import Counter
    emotion_counts = Counter([item['original_emotion'] for item in train_data])
    
    print("\n当前情绪分布:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}条")
    
    # 增强"支持"和"快乐"类
    print("\n" + "="*60)
    print("开始数据增强...")
    print("="*60)
    
    augmented_data = augment_emotion_data(
        train_data,
        target_emotions=['支持', '快乐'],
        target_count=80  # 每类补充到80条
    )
    
    # 保存增强后的数据
    output_path = str(_PROJECT_ROOT / "05_数据文件" / "training_set_hmtl_augmented.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    
    # 统计增强后分布
    augmented_counts = Counter([item['original_emotion'] for item in augmented_data])
    
    print("\n" + "="*60)
    print("增强后情绪分布:")
    print("="*60)
    for emotion in sorted(emotion_counts.keys()):
        before = emotion_counts[emotion]
        after = augmented_counts[emotion]
        change = after - before
        print(f"  {emotion}: {before}条 → {after}条 (+{change})")
    
    print(f"\n总样本数: {len(train_data)} → {len(augmented_data)} "
          f"(+{len(augmented_data) - len(train_data)})")
    
    print(f"\n✅ 增强数据已保存到: {output_path}")
    
    # 显示几个增强样本示例
    print("\n" + "="*60)
    print("增强样本示例")
    print("="*60)
    
    aug_samples = [item for item in augmented_data 
                   if item.get('source', '').startswith('data_augmentation')]
    
    for emotion in ['支持', '快乐']:
        print(f"\n{emotion}类增强示例:")
        emotion_augs = [s for s in aug_samples if s['original_emotion'] == emotion]
        for i, sample in enumerate(emotion_augs[:3], 1):
            print(f"  {i}. {sample['text']}")


if __name__ == "__main__":
    main()
