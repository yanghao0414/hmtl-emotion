#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL 多模态情绪识别系统 - Web可视化界面
使用Gradio构建交互式界面
"""

import sys
import os
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

bootstrap()

import gradio as gr
import numpy as np
from PIL import Image
from multimodal_fusion_v2 import MultimodalFusionSystemV2

# 全局系统实例
system = None

def initialize_system():
    """初始化系统"""
    global system
    if system is None:
        print("🔄 初始化多模态融合系统...")
        system = MultimodalFusionSystemV2()
        system.initialize_models(load_visual=True, load_audio=False)
    return system

def predict_emotion(text, image, strategy):
    """
    预测情绪
    Args:
        text: 输入文本
        image: 上传的图像 (PIL Image 或 None)
        strategy: 融合策略
    """
    global system
    
    if system is None:
        initialize_system()
    
    # 检查输入
    if not text and image is None:
        return "⚠️ 请输入文本或上传图像", "", "", "", ""
    
    # 执行预测
    try:
        result = system.fuse(
            text=text if text else None,
            image=image,
            strategy=strategy
        )
        
        if not result or 'fusion_result' not in result:
            return "❌ 预测失败", "", "", "", ""
        
        fusion = result['fusion_result']
        single = result.get('single_predictions', {})
        
        # 主结果
        main_result = f"""
## 🎯 融合结果

| 项目 | 结果 |
|------|------|
| **4类情绪** | {fusion['4类分类']} |
| **7类情绪** | {fusion['7类情绪']} |
| **3类极性** | {fusion['3类极性']} |
| **唤醒度** | {fusion['arousal']:.3f} |
| **效价** | {fusion['valence']:.3f} |
| **置信度** | {fusion['confidence']:.2%} |
| **融合策略** | {fusion['strategy']} |
| **使用模态** | {', '.join(fusion.get('modalities_used', []))} |
"""
        
        # 文本预测详情
        text_detail = ""
        if 'text' in single:
            t = single['text']
            text_detail = f"""
### 📝 文本模型预测
- **4类**: {t['4类分类']} (置信度: {t['confidence']:.2%})
- **7类**: {t['7类情绪']}
- **Arousal**: {t['arousal']:.3f}
- **Valence**: {t['valence']:.3f}
"""
        
        # 视觉预测详情
        visual_detail = ""
        if 'visual' in single:
            v = single['visual']
            visual_detail = f"""
### 👁️ 视觉模型预测
- **4类**: {v['4类分类']} (置信度: {v['confidence']:.2%})
- **7类**: {v['7类情绪']}
- **Arousal**: {v['arousal']:.3f}
- **Valence**: {v['valence']:.3f}
"""
        
        # 情绪解释
        emotion_explain = get_emotion_explanation(fusion['4类分类'], fusion['7类情绪'])
        
        # 置信度条
        confidence_bar = f"{fusion['confidence']:.1%}"
        
        return main_result, text_detail, visual_detail, emotion_explain, confidence_bar
        
    except Exception as e:
        return f"❌ 错误: {str(e)}", "", "", "", ""


def get_emotion_explanation(emotion_4, emotion_7):
    """获取情绪解释"""
    explanations = {
        '积极': '😊 **积极情绪**: 表示正面、愉快的情感状态，包括快乐、满足、支持等。',
        '激活消极': '😠 **激活消极情绪**: 高唤醒度的负面情绪，如愤怒、焦虑、恐惧等。',
        '非激活消极': '😢 **非激活消极情绪**: 低唤醒度的负面情绪，如悲伤、失望、沮丧等。',
        '非激活型消极': '😢 **非激活消极情绪**: 低唤醒度的负面情绪，如悲伤、失望、沮丧等。',
        '平静': '😌 **平静情绪**: 中性、平和的情感状态，没有明显的正负倾向。'
    }
    
    emotion_7_details = {
        '愤怒': '💢 愤怒 - 对不公或伤害的强烈反应',
        '焦虑': '😰 焦虑 - 对未来不确定性的担忧',
        '快乐': '😄 快乐 - 满足和愉悦的状态',
        '悲伤': '😭 悲伤 - 失去或失望带来的痛苦',
        '失望': '😞 失望 - 期望落空的感受',
        '支持': '🤝 支持 - 认同和鼓励的态度',
        '平静': '😌 平静 - 内心安宁的状态'
    }
    
    base = explanations.get(emotion_4, f"情绪类型: {emotion_4}")
    detail = emotion_7_details.get(emotion_7, f"具体情绪: {emotion_7}")
    
    return f"{base}\n\n**具体情绪**: {detail}"


def create_interface():
    """创建Gradio界面"""
    
    # 初始化系统
    initialize_system()
    
    with gr.Blocks(
        title="HMTL 多模态情绪识别系统",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .result-box { padding: 15px; border-radius: 10px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🎭 HMTL 多模态情绪识别系统
        
        支持**文本**、**图像**多模态输入，智能融合预测情绪。
        
        ---
        """)
        
        with gr.Row():
            # 左侧：输入区
            with gr.Column(scale=1):
                gr.Markdown("### 📥 输入")
                
                text_input = gr.Textbox(
                    label="文本输入",
                    placeholder="输入要分析的文本，例如：我今天心情很好...",
                    lines=3
                )
                
                image_input = gr.Image(
                    label="图像输入 (可选)",
                    type="pil",
                    height=200
                )
                
                strategy_input = gr.Radio(
                    choices=["weighted", "hierarchical", "confidence"],
                    value="weighted",
                    label="融合策略",
                    info="weighted=加权投票, hierarchical=层级融合, confidence=置信度加权"
                )
                
                predict_btn = gr.Button("🔍 分析情绪", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                ### 📊 模型信息
                | 模型 | 准确率 |
                |------|--------|
                | 文本 | ~65% |
                | 视觉 | ~70% |
                | 音频 | ~75% |
                """)
            
            # 右侧：结果区
            with gr.Column(scale=1):
                gr.Markdown("### 📤 预测结果")
                
                main_output = gr.Markdown(label="融合结果")
                
                with gr.Accordion("📝 文本模型详情", open=False):
                    text_output = gr.Markdown()
                
                with gr.Accordion("👁️ 视觉模型详情", open=False):
                    visual_output = gr.Markdown()
                
                with gr.Accordion("💡 情绪解释", open=True):
                    explain_output = gr.Markdown()
                
                confidence_output = gr.Textbox(label="置信度", interactive=False)
        
        # 示例
        gr.Markdown("---\n### 📌 示例")
        gr.Examples(
            examples=[
                ["我今天心情很好，工作顺利完成了！", None, "weighted"],
                ["感觉很失落和沮丧，什么都不想做", None, "weighted"],
                ["这件事让我非常生气！", None, "weighted"],
                ["今天天气不错，没什么特别的", None, "weighted"],
            ],
            inputs=[text_input, image_input, strategy_input],
            label="点击示例快速测试"
        )
        
        # 绑定事件
        predict_btn.click(
            fn=predict_emotion,
            inputs=[text_input, image_input, strategy_input],
            outputs=[main_output, text_output, visual_output, explain_output, confidence_output]
        )
        
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        1. **输入文本**: 在文本框中输入要分析的内容
        2. **上传图像** (可选): 上传人脸图像进行视觉情绪分析
        3. **选择融合策略**: 
           - `weighted`: 根据模型准确率加权投票
           - `hierarchical`: 层级融合 (先融合非语言模态)
           - `confidence`: 根据预测置信度动态加权
        4. **点击分析**: 查看多模态融合的情绪预测结果
        
        ---
        *HMTL Multi-modal Emotion Recognition System v2.0*
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # 启用公开分享链接
        inbrowser=True
    )
