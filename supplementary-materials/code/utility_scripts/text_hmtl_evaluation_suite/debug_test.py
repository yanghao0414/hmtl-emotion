#!/usr/bin/env python3
"""调试：检查为什么4类和7类准确率一样"""

import os
import sys
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '02_模型代码'))
from transformers import BertTokenizer
from hmtl_model_v2 import HMTLEmotionModelV2

EMOTION_7_NAMES = {0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤', 4: '失望', 5: '支持', 6: '平静'}
LABEL_4_NAMES = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}
LABEL_7_TO_4 = {0: 1, 1: 1, 2: 0, 3: 2, 4: 2, 5: 0, 6: 3}

# 加载模型
device = 'cpu'
model = HMTLEmotionModelV2()
checkpoint = torch.load(r"d:\bigcreate\06_模型文件\hmtl_models_v2\best_model_v2.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载数据
df = pd.read_csv("data/raw/evaluation_dataset.csv")
df_manual = df[df['source'] == 'manual'].copy()

# 逐条预测并对比
print("逐条对比 7类预测 vs 4类预测:")
print("="*80)

correct_7 = 0
correct_4 = 0
correct_4_from_7 = 0  # 从7类转换得到的4类

for idx, row in df_manual.iterrows():
    text = row['text']
    true_7 = int(row['label_7'])
    true_4 = LABEL_7_TO_4[true_7]
    
    # 预测
    encoding = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
    
    pred_7 = outputs['label_7_logits'].argmax(dim=1).item()
    pred_4 = outputs['label_4_logits'].argmax(dim=1).item()
    pred_4_from_7 = LABEL_7_TO_4[pred_7]  # 从7类预测转换
    
    # 统计
    if pred_7 == true_7:
        correct_7 += 1
    if pred_4 == true_4:
        correct_4 += 1
    if pred_4_from_7 == true_4:
        correct_4_from_7 += 1
    
    # 显示不一致的情况
    if pred_4 != pred_4_from_7:
        print(f"文本: {text[:30]}...")
        print(f"  真实: 7类={EMOTION_7_NAMES[true_7]}, 4类={LABEL_4_NAMES[true_4]}")
        print(f"  预测: 7类={EMOTION_7_NAMES[pred_7]}, 4类直接={LABEL_4_NAMES[pred_4]}, 4类从7类转={LABEL_4_NAMES[pred_4_from_7]}")
        print()

print("="*80)
print(f"7类准确率: {correct_7}/56 = {correct_7/56*100:.1f}%")
print(f"4类准确率(模型直接输出): {correct_4}/56 = {correct_4/56*100:.1f}%")
print(f"4类准确率(从7类转换): {correct_4_from_7}/56 = {correct_4_from_7/56*100:.1f}%")
