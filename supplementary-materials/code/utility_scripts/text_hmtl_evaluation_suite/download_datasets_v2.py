#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从公开数据集下载中文情绪文本 V2
使用更稳定的数据源
"""

import os
import json
import random
import pandas as pd
import urllib.request

# 情绪映射
EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}
LABEL_7_TO_3 = {0: 1, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2}
AV_TYPICAL = {
    0: (0.9, -0.8), 1: (0.7, -0.6), 2: (0.7, 0.9), 3: (0.3, -0.7),
    4: (0.4, -0.5), 5: (0.5, 0.7), 6: (0.2, 0.3),
}


def download_chnsenticorp():
    """下载 ChnSentiCorp 数据集（酒店评论）"""
    print("\n下载 ChnSentiCorp 数据集...")
    
    # 直接从 GitHub 下载
    urls = {
        'train': 'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv'
    }
    
    samples = []
    try:
        for name, url in urls.items():
            print(f"  下载 {name}...")
            response = urllib.request.urlopen(url, timeout=30)
            content = response.read().decode('utf-8')
            
            lines = content.strip().split('\n')
            for line in lines[1:]:  # 跳过表头
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    label, text = parts
                    try:
                        orig_label = int(label)
                        if len(text) >= 10:
                            # 映射: 0=负面, 1=正面
                            if orig_label == 0:
                                label_7 = random.choice([3, 4])  # 悲伤/失望
                            else:
                                label_7 = random.choice([2, 6])  # 快乐/平静
                            
                            samples.append({
                                'text': text[:200],
                                'label_7': label_7,
                                'source': 'chnsenticorp_hotel',
                                'needs_review': True
                            })
                    except:
                        continue
        
        print(f"  下载完成: {len(samples)} 条")
        
    except Exception as e:
        print(f"  下载失败: {e}")
    
    return samples


def download_waimai():
    """下载外卖评论数据集"""
    print("\n下载外卖评论数据集...")
    
    url = 'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv'
    
    samples = []
    try:
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode('utf-8')
        
        lines = content.strip().split('\n')
        for line in lines[1:]:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                label, text = parts
                try:
                    orig_label = int(label)
                    if len(text) >= 5:
                        # 映射: 0=负面, 1=正面
                        if orig_label == 0:
                            label_7 = random.choice([0, 4])  # 愤怒/失望
                        else:
                            label_7 = random.choice([2, 5])  # 快乐/支持
                        
                        samples.append({
                            'text': text[:200],
                            'label_7': label_7,
                            'source': 'waimai_review',
                            'needs_review': True
                        })
                except:
                    continue
        
        print(f"  下载完成: {len(samples)} 条")
        
    except Exception as e:
        print(f"  下载失败: {e}")
    
    return samples


def download_weibo_senti():
    """下载微博情感数据"""
    print("\n下载微博情感数据集...")
    
    url = 'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv'
    
    samples = []
    try:
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode('utf-8')
        
        lines = content.strip().split('\n')
        for line in lines[1:]:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                label, text = parts
                try:
                    orig_label = int(label)
                    if len(text) >= 10:
                        # 映射: 0=负面, 1=正面
                        if orig_label == 0:
                            label_7 = random.choice([0, 1, 3, 4])
                        else:
                            label_7 = random.choice([2, 5, 6])
                        
                        samples.append({
                            'text': text[:200],
                            'label_7': label_7,
                            'source': 'weibo_senti',
                            'needs_review': True
                        })
                except:
                    continue
        
        print(f"  下载完成: {len(samples)} 条")
        
    except Exception as e:
        print(f"  下载失败: {e}")
    
    return samples


def create_diverse_samples():
    """创建多样化的手动样本"""
    print("\n创建多样化样本...")
    
    samples = [
        # 愤怒 (0)
        {"text": "这服务态度也太差了吧！气死我了！", "label_7": 0, "source": "manual"},
        {"text": "什么破东西，完全是骗人的！", "label_7": 0, "source": "manual"},
        {"text": "太过分了，必须投诉！", "label_7": 0, "source": "manual"},
        {"text": "简直不能忍，怎么会有这种人！", "label_7": 0, "source": "manual"},
        {"text": "垃圾！浪费我的时间和钱！", "label_7": 0, "source": "manual"},
        {"text": "我要退款！这是欺诈！", "label_7": 0, "source": "manual"},
        {"text": "再也不会来了，太恶心了", "label_7": 0, "source": "manual"},
        {"text": "你们这是什么态度？！", "label_7": 0, "source": "manual"},
        
        # 焦虑 (1)
        {"text": "好担心明天的面试结果...", "label_7": 1, "source": "manual"},
        {"text": "不知道该怎么办才好，好迷茫", "label_7": 1, "source": "manual"},
        {"text": "越想越紧张，睡不着觉", "label_7": 1, "source": "manual"},
        {"text": "万一失败了怎么办？", "label_7": 1, "source": "manual"},
        {"text": "心里七上八下的，好不安", "label_7": 1, "source": "manual"},
        {"text": "压力好大，感觉喘不过气", "label_7": 1, "source": "manual"},
        {"text": "不确定这样做对不对...", "label_7": 1, "source": "manual"},
        {"text": "总觉得会出问题，好慌", "label_7": 1, "source": "manual"},
        
        # 快乐 (2)
        {"text": "太棒了！超级开心！", "label_7": 2, "source": "manual"},
        {"text": "哈哈哈笑死我了，太搞笑了", "label_7": 2, "source": "manual"},
        {"text": "今天运气真好，中奖了！", "label_7": 2, "source": "manual"},
        {"text": "终于成功了！激动！", "label_7": 2, "source": "manual"},
        {"text": "好幸福啊，感谢遇见你", "label_7": 2, "source": "manual"},
        {"text": "这也太好吃了吧！爱了爱了", "label_7": 2, "source": "manual"},
        {"text": "开心到飞起~今天是好日子", "label_7": 2, "source": "manual"},
        {"text": "收到礼物了！惊喜！", "label_7": 2, "source": "manual"},
        
        # 悲伤 (3)
        {"text": "心里好难受，想哭...", "label_7": 3, "source": "manual"},
        {"text": "眼泪止不住地流", "label_7": 3, "source": "manual"},
        {"text": "感觉好孤独，没人理解我", "label_7": 3, "source": "manual"},
        {"text": "再也回不去了，好怀念从前", "label_7": 3, "source": "manual"},
        {"text": "好想哭，心好痛", "label_7": 3, "source": "manual"},
        {"text": "失去了最重要的人...", "label_7": 3, "source": "manual"},
        {"text": "一个人的夜晚，好难熬", "label_7": 3, "source": "manual"},
        {"text": "心碎了，无法愈合", "label_7": 3, "source": "manual"},
        
        # 失望 (4)
        {"text": "真让人失望，不是说好的吗", "label_7": 4, "source": "manual"},
        {"text": "没想到会是这样的结果", "label_7": 4, "source": "manual"},
        {"text": "期望太高了，现实太骨感", "label_7": 4, "source": "manual"},
        {"text": "白等了这么久，唉", "label_7": 4, "source": "manual"},
        {"text": "算了，不抱希望了", "label_7": 4, "source": "manual"},
        {"text": "和想象中差太多了", "label_7": 4, "source": "manual"},
        {"text": "本来以为会很好，结果...", "label_7": 4, "source": "manual"},
        {"text": "又一次被放鸽子了", "label_7": 4, "source": "manual"},
        
        # 支持 (5)
        {"text": "加油，你可以的！相信自己", "label_7": 5, "source": "manual"},
        {"text": "我理解你的感受，别担心", "label_7": 5, "source": "manual"},
        {"text": "别担心，会好起来的", "label_7": 5, "source": "manual"},
        {"text": "有什么需要帮忙的吗？", "label_7": 5, "source": "manual"},
        {"text": "你做得很好，继续保持", "label_7": 5, "source": "manual"},
        {"text": "我支持你的决定", "label_7": 5, "source": "manual"},
        {"text": "没关系，下次会更好的", "label_7": 5, "source": "manual"},
        {"text": "你不是一个人，我陪你", "label_7": 5, "source": "manual"},
        
        # 平静 (6)
        {"text": "还行吧，一般般", "label_7": 6, "source": "manual"},
        {"text": "正常操作，没什么特别的", "label_7": 6, "source": "manual"},
        {"text": "知道了，收到", "label_7": 6, "source": "manual"},
        {"text": "好的，我看看", "label_7": 6, "source": "manual"},
        {"text": "就这样吧，无所谓", "label_7": 6, "source": "manual"},
        {"text": "嗯，了解了", "label_7": 6, "source": "manual"},
        {"text": "可以，没问题", "label_7": 6, "source": "manual"},
        {"text": "随便吧，都行", "label_7": 6, "source": "manual"},
    ]
    
    for s in samples:
        s['needs_review'] = False
    
    print(f"  创建完成: {len(samples)} 条")
    return samples


def save_dataset(samples: list, output_path: str, max_per_source: int = 100):
    """保存数据集"""
    # 按来源采样
    by_source = {}
    for s in samples:
        src = s['source']
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(s)
    
    final_samples = []
    for src, items in by_source.items():
        if src == 'manual':
            final_samples.extend(items)  # 手动样本全部保留
        else:
            final_samples.extend(random.sample(items, min(max_per_source, len(items))))
    
    # 转换为 DataFrame
    data = []
    for i, s in enumerate(final_samples):
        label_7 = s['label_7']
        a, v = AV_TYPICAL.get(label_7, (0.5, 0))
        
        data.append({
            'id': f"eval_{i:04d}",
            'text': s['text'],
            'source': s['source'],
            'label_7': label_7 if not s.get('needs_review') else '',
            'label_7_name': EMOTION_7_NAMES.get(label_7, '') if not s.get('needs_review') else '',
            'label_4': LABEL_7_TO_4.get(label_7, '') if not s.get('needs_review') else '',
            'label_3': LABEL_7_TO_3.get(label_7, '') if not s.get('needs_review') else '',
            'arousal': a if not s.get('needs_review') else '',
            'valence': v if not s.get('needs_review') else '',
            'annotator': 'auto' if not s.get('needs_review') else '',
            'confidence': 'high' if not s.get('needs_review') else '',
            'notes': '需要人工校验' if s.get('needs_review') else ''
        })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig 方便 Excel 打开
    
    print(f"\n已保存到: {output_path}")
    print(f"总计 {len(data)} 条")
    
    print("\n来源分布:")
    for source in df['source'].unique():
        count = len(df[df['source'] == source])
        print(f"  {source}: {count}")
    
    # 统计需要校验的
    needs_review = len(df[df['notes'] == '需要人工校验'])
    print(f"\n需要人工校验: {needs_review} 条")
    print(f"已标注完成: {len(data) - needs_review} 条")


def main():
    print("="*60)
    print("中文情绪数据收集工具 V2")
    print("="*60)
    
    all_samples = []
    
    # 1. 手动样本（已标注，无需校验）
    manual = create_diverse_samples()
    all_samples.extend(manual)
    
    # 2. 下载公开数据集
    weibo = download_weibo_senti()
    all_samples.extend(weibo)
    
    hotel = download_chnsenticorp()
    all_samples.extend(hotel)
    
    waimai = download_waimai()
    all_samples.extend(waimai)
    
    # 3. 保存
    save_dataset(all_samples, "data/raw/evaluation_dataset.csv", max_per_source=150)
    
    print("\n" + "="*60)
    print("下一步操作:")
    print("="*60)
    print("1. 用 Excel 打开 data/raw/evaluation_dataset.csv")
    print("2. 找到 '需要人工校验' 的行，检查并修正 label_7")
    print("3. 填写 arousal (0-1) 和 valence (-1到1)")
    print("4. 保存到 data/annotated/evaluation_dataset_annotated.csv")
    print("5. 运行评估:")
    print("   python run_evaluation.py --data data/annotated/evaluation_dataset_annotated.csv")
    print("="*60)


if __name__ == '__main__':
    main()
