"""
Audio HMTL 音频情绪测试工具
支持单文件和文件夹批量测试
"""

import torch
import librosa
import os
import sys

# 导入依赖
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules', 'audio_hmtl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from transformers import Wav2Vec2Processor
from audio_hmtl_classifier import AudioHMTLClassifier

# --- 配置 ---
MODEL_PATH = "06_模型文件/audio_best_hmtl.pt"
PROCESSOR_NAME = "facebook/wav2vec2-base"
MAX_SEQ_LEN = 48000  # 3秒音频

# 标签名称
EMOTION_LABELS = {
    0: "积极  (Happy)",
    1: "激活消极  (Sad)", 
    2: "非激活消极  (Angry)",
    3: "平静  (Neutral)"
}

POLARITY_LABELS = {
    0: "积极 (Positive)",
    1: "消极 (Negative)",
    2: "平静 (Neutral)"
}


def load_model():
    """加载模型"""
    print("加载模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")
    
    model = AudioHMTLClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    
    print("模型加载完成")
    return model, processor, device


def predict_emotion(audio_path, model, processor, device):
    """预测单条音频情绪"""
    
    # 加载音频
    speech, rate = librosa.load(audio_path, sr=16000)
    
    # 截断过长音频
    if len(speech) > MAX_SEQ_LEN:
        speech = speech[:MAX_SEQ_LEN]
    
    # 特征提取
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = torch.ones(input_values.shape, dtype=torch.long).to(device)
    
    # 推理
    with torch.no_grad():
        logits_4, logits_3, pred_A, pred_V = model(input_values, attention_mask)
    
    # 解析结果
    emotion_4 = torch.argmax(logits_4, dim=1).item()
    emotion_3 = torch.argmax(logits_3, dim=1).item()
    arousal = pred_A.item()
    valence = pred_V.item()
    
    # 置信度
    probs_4 = torch.softmax(logits_4, dim=1)[0]
    confidence = probs_4[emotion_4].item()
    
    return {
        'emotion_4': emotion_4,
        'emotion_4_label': EMOTION_LABELS[emotion_4],
        'confidence': confidence,
        'polarity': POLARITY_LABELS[emotion_3],
        'arousal': arousal,
        'valence': valence,
        'all_probs': {EMOTION_LABELS[i]: f"{probs_4[i].item():.2%}" for i in range(4)}
    }


def test_single_audio(audio_path):
    """测试单个音频文件"""
    
    if not os.path.exists(audio_path):
        print(f"文件不存在: {audio_path}")
        return
    
    model, processor, device = load_model()
    
    print(f"\n测试文件: {os.path.basename(audio_path)}")
    print("=" * 50)
    
    result = predict_emotion(audio_path, model, processor, device)
    
    print(f"\n预测结果:")
    print(f"  情绪类别: {result['emotion_4_label']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  极性: {result['polarity']}")
    print(f"  Arousal: {result['arousal']:.3f}")
    print(f"  Valence: {result['valence']:.3f}")
    
    print(f"\n各类概率:")
    for label, prob in result['all_probs'].items():
        print(f"   {label}: {prob}")
    
    return result


def test_folder(folder_path):
    """测试文件夹中所有音频"""
    
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return
    
    model, processor, device = load_model()
    
    # 扫描音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not audio_files:
        print(f"未找到音频文件")
        return
    
    print(f"\n找到 {len(audio_files)} 个音频文件")
    print("=" * 60)
    
    results = []
    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        result = predict_emotion(audio_path, model, processor, device)
        result['filename'] = audio_file
        results.append(result)
        
        print(f"\n{audio_file}:")
        print(f"   → {result['emotion_4_label']} ({result['confidence']:.1%})")
    
    # 统计
    print("\n" + "=" * 60)
    print("统计结果:")
    emotion_counts = {}
    for r in results:
        label = r['emotion_4_label']
        emotion_counts[label] = emotion_counts.get(label, 0) + 1
    
    for label, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} 个 ({count/len(results):.1%})")
    
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Audio HMTL 音频情绪测试")
    print("=" * 60)
    
    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"模型不存在: {MODEL_PATH}")
        print("请先训练模型")
        sys.exit(1)
    
    # 菜单
    print("\n选择测试模式:")
    print("  1. 单文件测试")
    print("  2. 文件夹批量测试")
    print("  3. 录音测试")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        path = input("音频路径: ").strip().strip('"')
        test_single_audio(path)
    elif choice == '2':
        path = input("文件夹路径: ").strip().strip('"')
        test_folder(path)
    elif choice == '3':
        path = input("音频路径: ").strip().strip('"')
        test_single_audio(path)
    else:
        # 显示用法
        print("\n用法示例:")
        print('  python test_audio_model.py')
        print('  然后选择测试模式')
