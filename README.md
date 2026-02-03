---
title: HMTL 多模态情绪识别系统
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 HMTL 多模态情绪识别系统

基于深度学习的中文情绪识别系统。

## 功能

- **4类情绪分类**: 积极、激活消极、非激活消极、平静
- **7类情绪分类**: 愤怒、焦虑、快乐、悲伤、失望、支持、平静
- **情绪维度**: 唤醒度 (Arousal) 和 效价 (Valence)

## 使用方法

1. 在文本框中输入要分析的中文文本
2. 点击"分析情绪"按钮
3. 查看预测结果

## 示例

- "我今天心情很好" → 积极
- "感觉很失落" → 非激活消极
- "这让我很生气" → 激活消极
- "没什么特别的" → 平静
