#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL多模态情绪预测 - 完整版
支持: 文本 + 图像 + 音频
"""

import os

from path_bootstrap import bootstrap

bootstrap()

import numpy as np
from PIL import Image
from multimodal_fusion_v2 import MultimodalFusionSystemV2


def load_image(image_path):
    """加载图像"""
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    try:
        img = Image.open(image_path).convert('RGB')
        print(f"✅ 图像已加载: {image_path}")
        return img
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        return None


def load_audio(audio_path):
    """加载音频"""
    if not os.path.exists(audio_path):
        print(f"❌ 音频文件不存在: {audio_path}")
        return None
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"✅ 音频已加载: {audio_path} ({len(audio)/sr:.2f}秒)")
        return audio
    except ImportError:
        print("⚠️ librosa未安装，尝试使用scipy...")
        try:
            from scipy.io import wavfile
            sr, audio = wavfile.read(audio_path)
            audio = audio.astype(np.float32) / 32768.0
            print(f"✅ 音频已加载: {audio_path}")
            return audio
        except Exception as e:
            print(f"❌ 音频加载失败: {e}")
            return None
    except Exception as e:
        print(f"❌ 音频加载失败: {e}")
        return None


def print_result(result):
    """打印预测结果"""
    if not result:
        print("❌ 无预测结果")
        return
    
    print("\n" + "="*50)
    
    # 单模态结果
    if 'single_predictions' in result:
        print("📊 各模态预测结果:")
        for modality, pred in result['single_predictions'].items():
            print(f"\n  【{modality.upper()}】")
            print(f"    4类分类: {pred['4类分类']}")
            print(f"    7类情绪: {pred['7类情绪']}")
            print(f"    置信度:  {pred.get('confidence', 0):.3f}")
    
    # 融合结果
    if 'fusion_result' in result:
        r = result['fusion_result']
        print(f"\n🔗 融合结果 ({r.get('strategy', 'unknown')}):")
        print(f"    4类分类: {r['4类分类']}")
        print(f"    7类情绪: {r['7类情绪']}")
        print(f"    3类极性: {r['3类极性']}")
        print(f"    唤醒度:  {r['arousal']:.3f}")
        print(f"    效价:    {r['valence']:.3f}")
        print(f"    使用模态: {r.get('modalities_used', [])}")
    
    print("="*50)


def interactive_mode(system):
    """交互模式"""
    print("\n" + "-"*50)
    print("📌 交互模式 - 命令说明:")
    print("  输入文本直接预测")
    print("  image:路径  - 添加图像")
    print("  audio:路径  - 添加音频")
    print("  predict     - 执行多模态预测")
    print("  clear       - 清除当前输入")
    print("  quit        - 退出")
    print("-"*50)
    
    current_text = None
    current_image = None
    current_audio = None
    
    while True:
        try:
            # 显示当前状态
            status = []
            if current_text:
                status.append(f"文本: '{current_text[:20]}...'")
            if current_image is not None:
                status.append("图像: ✓")
            if current_audio is not None:
                status.append("音频: ✓")
            
            if status:
                print(f"\n📋 当前输入: {', '.join(status)}")
            
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            # 退出
            if cmd.lower() in ['quit', 'q', 'exit', '退出']:
                print("\n👋 再见！")
                break
            
            # 清除
            if cmd.lower() == 'clear':
                current_text = None
                current_image = None
                current_audio = None
                print("✅ 已清除所有输入")
                continue
            
            # 添加图像
            if cmd.lower().startswith('image:'):
                path = cmd[6:].strip()
                current_image = load_image(path)
                continue
            
            # 添加音频
            if cmd.lower().startswith('audio:'):
                path = cmd[6:].strip()
                current_audio = load_audio(path)
                continue
            
            # 执行预测
            if cmd.lower() == 'predict':
                if not current_text and current_image is None and current_audio is None:
                    print("⚠️ 请先输入文本、图像或音频")
                    continue
                
                result = system.fuse(
                    text=current_text,
                    image=current_image,
                    audio=current_audio,
                    strategy='weighted'
                )
                print_result(result)
                continue
            
            # 默认作为文本输入并立即预测
            current_text = cmd
            
            # 如果只有文本，直接预测
            if current_image is None and current_audio is None:
                result = system.fuse(text=current_text, strategy='weighted')
                print_result(result)
            else:
                # 有其他模态，等待predict命令
                print(f"✅ 文本已设置: '{current_text[:30]}...'")
                print("💡 输入 'predict' 执行多模态融合预测")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


def demo_mode(system):
    """演示模式 - 使用模拟数据"""
    print("\n🎬 演示模式 - 使用模拟数据")
    print("="*50)
    
    test_cases = [
        {"text": "我今天心情很好，工作进展顺利", "desc": "积极情绪"},
        {"text": "我很担心明天的考试，压力很大", "desc": "焦虑情绪"},
        {"text": "感觉很失落和沮丧", "desc": "悲伤情绪"},
        {"text": "心情很平静，没什么特别的", "desc": "平静情绪"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n【测试 {i}: {case['desc']}】")
        print(f"文本: {case['text']}")
        
        # 创建模拟图像和音频
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        mock_audio = np.random.randn(16000)  # 1秒
        
        result = system.fuse(
            text=case['text'],
            image=mock_image,
            audio=mock_audio,
            strategy='weighted'
        )
        print_result(result)
    
    print("\n✅ 演示完成！")
    print("⚠️ 注意: 图像和音频使用随机数据，真实应用需要真实输入")


def main():
    print("="*60)
    print("🎯 HMTL多模态情绪识别系统 - 完整版")
    print("   支持: 文本 + 图像 + 音频")
    print("="*60)
    
    # 初始化系统
    print("\n🔄 正在加载所有模型...")
    system = MultimodalFusionSystemV2()
    system.initialize_models(load_visual=True, load_audio=True)
    
    print("\n✅ 系统就绪！")
    
    # 选择模式
    print("\n请选择模式:")
    print("  1. 交互模式 - 手动输入文本/图像/音频")
    print("  2. 演示模式 - 使用模拟数据演示")
    print("  3. 退出")
    
    while True:
        choice = input("\n请选择 (1/2/3): ").strip()
        
        if choice == '1':
            interactive_mode(system)
            break
        elif choice == '2':
            demo_mode(system)
            break
        elif choice == '3':
            print("👋 再见！")
            break
        else:
            print("⚠️ 请输入 1, 2 或 3")


if __name__ == "__main__":
    main()
