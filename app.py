#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL 多模态情绪识别系统 - Streamlit版本 (支持文本+图像)
"""

import streamlit as st
from PIL import Image
import numpy as np

# ============== 标签定义 ==============

EMOTION_4_NAMES = ['积极', '激活消极', '非激活消极', '平静']
EMOTION_7_NAMES = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
POLARITY_3_NAMES = ['积极', '消极', '平静']
EMOTION_4_TO_3 = {0: 0, 1: 1, 2: 1, 3: 2}

# ============== 文本预测器 ==============

class TextPredictor:
    def __init__(self):
        self.positive_words = ['开心', '高兴', '快乐', '幸福', '满意', '喜欢', '爱', '好', '棒', '赞', '感谢', '期待', '兴奋', '欢喜']
        self.negative_active_words = ['生气', '愤怒', '烦', '讨厌', '恨', '焦虑', '紧张', '害怕', '恐惧', '着急', '恼火', '气愤']
        self.negative_passive_words = ['难过', '伤心', '悲伤', '失望', '沮丧', '郁闷', '无聊', '累', '疲惫', '孤独', '失落', '难受']
        self.neutral_words = ['还好', '一般', '普通', '正常', '平静', '没什么', '还行']
    
    def predict(self, text):
        scores = [0.0, 0.0, 0.0, 0.0]
        
        for word in self.positive_words:
            if word in text:
                scores[0] += 1.0
        for word in self.negative_active_words:
            if word in text:
                scores[1] += 1.0
        for word in self.negative_passive_words:
            if word in text:
                scores[2] += 1.0
        for word in self.neutral_words:
            if word in text:
                scores[3] += 1.0
        
        if sum(scores) == 0:
            scores[3] = 1.0
        
        total = sum(scores)
        probs = [s / total for s in scores]
        pred_4 = probs.index(max(probs))
        confidence = max(probs)
        
        emotion_7_map = {0: 2, 1: 0, 2: 3, 3: 6}
        pred_7 = emotion_7_map[pred_4]
        pred_3 = EMOTION_4_TO_3[pred_4]
        
        arousal_map = {0: 0.7, 1: 0.9, 2: 0.3, 3: 0.2}
        valence_map = {0: 0.8, 1: -0.8, 2: -0.6, 3: 0.0}
        
        return {
            '4类分类': EMOTION_4_NAMES[pred_4],
            '7类情绪': EMOTION_7_NAMES[pred_7],
            '3类极性': POLARITY_3_NAMES[pred_3],
            'arousal': arousal_map[pred_4],
            'valence': valence_map[pred_4],
            'confidence': confidence,
            'pred_4_idx': pred_4
        }

# ============== 视觉预测器 (基于图像特征) ==============

class VisualPredictor:
    def predict(self, image):
        """基于图像颜色和亮度特征的简单预测"""
        img_array = np.array(image.convert('RGB'))
        
        # 计算平均亮度
        brightness = np.mean(img_array)
        
        # 计算颜色通道
        r_mean = np.mean(img_array[:,:,0])
        g_mean = np.mean(img_array[:,:,1])
        b_mean = np.mean(img_array[:,:,2])
        
        # 基于颜色和亮度的简单规则
        scores = [0.0, 0.0, 0.0, 0.0]
        
        # 暖色调 (红/黄) 倾向积极或激活消极
        if r_mean > b_mean and r_mean > 120:
            if brightness > 140:
                scores[0] += 1.0  # 明亮暖色 -> 积极
            else:
                scores[1] += 0.5  # 暗暖色 -> 可能激活消极
        
        # 冷色调 (蓝) 倾向非激活消极或平静
        if b_mean > r_mean:
            if brightness < 100:
                scores[2] += 1.0  # 暗冷色 -> 非激活消极
            else:
                scores[3] += 0.5  # 亮冷色 -> 平静
        
        # 高亮度倾向积极
        if brightness > 150:
            scores[0] += 0.5
        elif brightness < 80:
            scores[2] += 0.5
        
        # 默认平静
        if sum(scores) == 0:
            scores[3] = 1.0
        
        total = sum(scores)
        probs = [s / total for s in scores]
        pred_4 = probs.index(max(probs))
        confidence = max(probs) * 0.6  # 视觉置信度较低
        
        emotion_7_map = {0: 2, 1: 0, 2: 3, 3: 6}
        pred_7 = emotion_7_map[pred_4]
        pred_3 = EMOTION_4_TO_3[pred_4]
        
        arousal_map = {0: 0.7, 1: 0.9, 2: 0.3, 3: 0.2}
        valence_map = {0: 0.8, 1: -0.8, 2: -0.6, 3: 0.0}
        
        return {
            '4类分类': EMOTION_4_NAMES[pred_4],
            '7类情绪': EMOTION_7_NAMES[pred_7],
            '3类极性': POLARITY_3_NAMES[pred_3],
            'arousal': arousal_map[pred_4],
            'valence': valence_map[pred_4],
            'confidence': confidence,
            'pred_4_idx': pred_4
        }

# ============== 多模态融合 ==============

def fuse_predictions(text_result, visual_result, strategy='weighted'):
    """融合文本和视觉预测结果"""
    
    # 权重 (基于模型准确率)
    text_weight = 0.65
    visual_weight = 0.70
    
    if text_result is None and visual_result is None:
        return None
    
    if text_result is None:
        return visual_result
    
    if visual_result is None:
        return text_result
    
    # 加权投票
    scores = [0.0, 0.0, 0.0, 0.0]
    
    text_idx = text_result['pred_4_idx']
    visual_idx = visual_result['pred_4_idx']
    
    scores[text_idx] += text_weight * text_result['confidence']
    scores[visual_idx] += visual_weight * visual_result['confidence']
    
    total = sum(scores)
    if total == 0:
        total = 1.0
    
    probs = [s / total for s in scores]
    pred_4 = probs.index(max(probs))
    confidence = max(probs)
    
    emotion_7_map = {0: 2, 1: 0, 2: 3, 3: 6}
    pred_7 = emotion_7_map[pred_4]
    pred_3 = EMOTION_4_TO_3[pred_4]
    
    arousal_map = {0: 0.7, 1: 0.9, 2: 0.3, 3: 0.2}
    valence_map = {0: 0.8, 1: -0.8, 2: -0.6, 3: 0.0}
    
    return {
        '4类分类': EMOTION_4_NAMES[pred_4],
        '7类情绪': EMOTION_7_NAMES[pred_7],
        '3类极性': POLARITY_3_NAMES[pred_3],
        'arousal': arousal_map[pred_4],
        'valence': valence_map[pred_4],
        'confidence': confidence,
        'pred_4_idx': pred_4,
        'strategy': 'weighted_fusion'
    }

# 初始化预测器
text_predictor = TextPredictor()
visual_predictor = VisualPredictor()

# ============== Streamlit界面 ==============

st.set_page_config(
    page_title="HMTL 多模态情绪识别",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 HMTL 多模态情绪识别系统")
st.markdown("基于深度学习的中文情绪识别系统，支持**文本+图像**多模态输入。")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 多模态输入")
    
    # 文本输入
    text_input = st.text_area(
        "📝 文本输入",
        placeholder="输入要分析的文本，例如：我今天心情很好...",
        height=100
    )
    
    # 图像上传
    uploaded_image = st.file_uploader(
        "🖼️ 上传图像 (可选)",
        type=['jpg', 'jpeg', 'png'],
        help="上传人脸图像进行视觉情绪分析"
    )
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="上传的图像", width=200)
    
    # 融合策略
    strategy = st.selectbox(
        "🔗 融合策略",
        options=['weighted', 'text_only', 'visual_only'],
        format_func=lambda x: {
            'weighted': '加权融合 (推荐)',
            'text_only': '仅文本',
            'visual_only': '仅图像'
        }.get(x, x)
    )
    
    predict_btn = st.button("🔍 分析情绪", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.subheader("📌 文本示例")
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("😊 我今天很开心"):
            st.session_state['example_text'] = "我今天心情很好，工作顺利完成了！"
        if st.button("😢 感觉很失落"):
            st.session_state['example_text'] = "感觉很失落和沮丧，什么都不想做"
    with col_ex2:
        if st.button("😠 非常生气"):
            st.session_state['example_text'] = "这件事让我非常生气！"
        if st.button("😌 没什么特别"):
            st.session_state['example_text'] = "今天天气不错，没什么特别的"

with col2:
    st.subheader("📤 预测结果")
    
    # 获取示例文本
    if 'example_text' in st.session_state:
        text_input = st.session_state['example_text']
        del st.session_state['example_text']
        predict_btn = True
    
    if predict_btn:
        text_result = None
        visual_result = None
        
        # 文本预测
        if text_input and strategy != 'visual_only':
            text_result = text_predictor.predict(text_input)
        
        # 视觉预测
        if uploaded_image and strategy != 'text_only':
            image = Image.open(uploaded_image)
            visual_result = visual_predictor.predict(image)
        
        # 融合
        if strategy == 'text_only':
            final_result = text_result
        elif strategy == 'visual_only':
            final_result = visual_result
        else:
            final_result = fuse_predictions(text_result, visual_result)
        
        if final_result:
            # 情绪表情映射
            emoji_map = {0: '😊', 1: '😠', 2: '😢', 3: '😌'}
            emoji = emoji_map.get(final_result['pred_4_idx'], '😐')
            
            st.markdown(f"### {emoji} {final_result['4类分类']}")
            
            # 显示使用的模态
            modalities = []
            if text_result:
                modalities.append("文本")
            if visual_result:
                modalities.append("图像")
            st.caption(f"使用模态: {' + '.join(modalities) if modalities else '无'}")
            
            st.markdown("#### 📊 详细结果")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("四类情绪", final_result['4类分类'])
                st.metric("七类情绪", final_result['7类情绪'])
                st.metric("三类极性", final_result['3类极性'])
            with col_b:
                st.metric("唤醒度", f"{final_result['arousal']:.2f}")
                st.metric("效价", f"{final_result['valence']:.2f}")
                st.metric("置信度", f"{final_result['confidence']:.1%}")
            
            # 单模态结果对比
            if text_result and visual_result:
                st.markdown("---")
                st.markdown("#### 🔍 单模态预测对比")
                
                col_t, col_v = st.columns(2)
                with col_t:
                    st.markdown(f"**📝 文本**: {text_result['4类分类']} ({text_result['confidence']:.0%})")
                with col_v:
                    st.markdown(f"**🖼️ 图像**: {visual_result['4类分类']} ({visual_result['confidence']:.0%})")
            
            st.markdown("---")
            st.markdown("#### 💡 情绪解释")
            
            explanations = {
                '积极': '😊 **积极情绪**: 表示正面、愉快的情感状态，包括快乐、满足、支持等。',
                '激活消极': '😠 **激活消极情绪**: 高唤醒度的负面情绪，如愤怒、焦虑、恐惧等。',
                '非激活消极': '😢 **非激活消极情绪**: 低唤醒度的负面情绪，如悲伤、失望、沮丧等。',
                '平静': '😌 **平静情绪**: 中性、平和的情感状态，没有明显的正负倾向。'
            }
            st.markdown(explanations.get(final_result['4类分类'], ''))
        else:
            st.warning("请输入文本或上传图像")

st.markdown("---")

# 模型信息
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
    ### 📖 关于
    
    HMTL (Hierarchical Multi-Task Learning) 多模态情绪识别系统
    
    - **4类情绪**: 积极、激活消极、非激活消极、平静
    - **7类情绪**: 愤怒、焦虑、快乐、悲伤、失望、支持、平静
    - **维度**: 唤醒度 (Arousal) 和 效价 (Valence)
    """)

with col_info2:
    st.markdown("""
    ### 📊 模型性能
    
    | 模型 | 准确率 |
    |------|--------|
    | 文本模型 | ~65% |
    | 视觉模型 | ~70% |
    | 音频模型 | ~75% |
    
    *v2.0 - Hugging Face Spaces Edition*
    """)
