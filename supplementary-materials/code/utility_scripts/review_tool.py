#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人工复核工具 - 快速复核AI预标注结果
支持批量确认、快速修改
"""

import json
import os

EMOTIONS = ["愤怒", "快乐", "焦虑", "悲伤", "失望", "支持", "平静"]

def review_samples(samples, emotion_type):
    """交互式复核样本"""
    print(f"\n{'='*60}")
    print(f"复核 {emotion_type} 样本 (共{len(samples)}条)")
    print(f"{'='*60}")
    print("\n操作说明:")
    print("  回车/y - 确认正确")
    print("  1-7   - 修改为其他情绪 (1=愤怒 2=快乐 3=焦虑 4=悲伤 5=失望 6=支持 7=平静)")
    print("  n     - 删除此样本")
    print("  q     - 完成复核")
    print()
    
    reviewed = []
    
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {sample['text'][:80]}")
        print(f"预测: {sample['main_emotion']} | 置信度: {sample['confidence']}")
        print(f"匹配: {', '.join(sample['matched_keywords'][:3])}")
        
        choice = input(f"确认? [回车=是]: ").strip().lower()
        
        if choice == 'q':
            print("已完成复核")
            break
        elif choice == 'n':
            print("✗ 已删除")
            continue
        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            idx = int(choice) - 1
            new_emotion = EMOTIONS[idx]
            sample['main_emotion'] = new_emotion
            sample['reviewed'] = True
            sample['modified'] = True
            reviewed.append(sample)
            print(f"✓ 已修改为: {new_emotion}")
        else:  # 回车或y
            sample['reviewed'] = True
            sample['modified'] = False
            reviewed.append(sample)
            print("✓ 已确认")
    
    return reviewed


def batch_confirm(samples, emotion_type):
    """批量确认高置信度样本"""
    print(f"\n{'='*60}")
    print(f"批量确认 {emotion_type} 样本")
    print(f"{'='*60}")
    
    high_conf = [s for s in samples if s['confidence'] >= 0.8]
    low_conf = [s for s in samples if s['confidence'] < 0.8]
    
    print(f"\n高置信度样本 (>= 0.8): {len(high_conf)}条")
    print(f"低置信度样本 (< 0.8): {len(low_conf)}条")
    
    # 显示几个高置信度示例
    print("\n高置信度样本示例:")
    for i, s in enumerate(high_conf[:3], 1):
        print(f"  {i}. {s['text'][:50]}")
    
    choice = input(f"\n批量确认 {len(high_conf)} 条高置信度样本? [y/n]: ").strip().lower()
    
    confirmed = []
    if choice == 'y':
        for s in high_conf:
            s['reviewed'] = True
            s['modified'] = False
            confirmed.append(s)
        print(f"✓ 已批量确认 {len(high_conf)} 条")
    
    # 逐个复核低置信度样本
    if low_conf:
        print(f"\n开始复核 {len(low_conf)} 条低置信度样本...")
        reviewed_low = review_samples(low_conf, emotion_type)
        confirmed.extend(reviewed_low)
    
    return confirmed


def main():
    """主函数"""
    print("="*60)
    print("人工复核工具")
    print("="*60)
    
    # 加载预标注结果
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    with open(str(_PROJECT_ROOT / "05_数据文件" / "auto_labeled_candidates.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n预标注统计:")
    print(f"  愤怒候选: {len(data['anger_samples'])}条")
    print(f"  快乐候选: {len(data['joy_samples'])}条")
    
    # 选择复核哪个类别
    print("\n请选择要复核的类别:")
    print("  1 - 愤怒")
    print("  2 - 快乐")
    print("  3 - 全部")
    
    choice = input("选择 [1-3]: ").strip()
    
    reviewed_all = []
    
    if choice in ['1', '3']:
        anger_reviewed = batch_confirm(data['anger_samples'], "愤怒")
        reviewed_all.extend(anger_reviewed)
    
    if choice in ['2', '3']:
        joy_reviewed = batch_confirm(data['joy_samples'], "快乐")
        reviewed_all.extend(joy_reviewed)
    
    # 保存复核结果
    if reviewed_all:
        output = {
            "total_reviewed": len(reviewed_all),
            "modified_count": sum(1 for s in reviewed_all if s.get('modified', False)),
            "samples": reviewed_all
        }
        
        output_file = str(_PROJECT_ROOT / "05_数据文件" / "reviewed_samples.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ 复核完成！")
        print(f"{'='*60}")
        print(f"复核样本数: {len(reviewed_all)}")
        print(f"修改样本数: {sum(1 for s in reviewed_all if s.get('modified', False))}")
        print(f"保存位置: {output_file}")
        
        # 统计分布
        from collections import Counter
        dist = Counter([s['main_emotion'] for s in reviewed_all])
        print("\n复核后分布:")
        for emo, count in dist.most_common():
            print(f"  {emo}: {count}条")
    else:
        print("\n未复核任何样本")


if __name__ == "__main__":
    main()
