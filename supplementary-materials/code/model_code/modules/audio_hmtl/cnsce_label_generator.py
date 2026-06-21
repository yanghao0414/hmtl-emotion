import os
import re
import pandas as pd
import numpy as np
import sys
import random

# 导入 HMTL 标签映射工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
try:
    from label_mapper import get_hmtl_labels, LABEL_3_MAP
except ImportError:
    print("无法导入 utils/label_mapper.py")
    sys.exit(1)

# =========================================================================
# TODO: 配置 CNSCED 数据集路径
# =========================================================================
CNSCED_ROOT = "D:/bigcreate/音频数据集/" 
OUTPUT_CSV = "05_数据集/audio_hmtl_labels.csv"

# CNSCED 情绪编码到 HMTL 7类情绪的映射 (核心映射表)
CNSCED_TO_HMTL_7 = {
    'A': '愤怒', 'S': '悲伤', 'H': '快乐', 
    'F': '焦虑', # Fear (恐惧) 映射为焦虑
    'B': '快乐', # Surprised (惊喜) 映射为快乐 (正面惊喜)
    '0': '平静', # Neutral (中性) 映射为平静
}

# 强度1, 2, 3 -> 0.0-1.0 的映射 (用于生成 Arousal/Valence 标签)
INTENSITY_MAP = {1: 0.3, 2: 0.6, 3: 0.9}


def parse_filename(filename):
    """
    解析 CNSCED 文件名生成 HMTL 标签和 A/V 值
    """
    emotion_parts = re.findall(r'([A-Z0])(\d)', filename)
    
    # 默认情绪标签
    default_hmtl = get_hmtl_labels('平静')
    if not emotion_parts:
        return {
            'label_4_emotion': default_hmtl['label_4'],
            'label_3_polarity': default_hmtl['label_3'],
            'true_arousal': 0.2, 
            'true_valence': 0.0, 
            'main_emotion_name': '平静'
        }

    # 1. 找到主要情绪
    max_intensity = 0
    main_hmtl_emotion = '平静' 
    
    for code, intensity_str in emotion_parts:
        intensity = int(intensity_str)
        
        # Arousal: 取最大强度
        max_intensity = max(max_intensity, intensity)
        
        #  W 
        current_hmtl_name = CNSCED_TO_HMTL_7.get(code)
        
        if intensity >= max_intensity:
            if code != 'W' and current_hmtl_name:
                 main_hmtl_emotion = current_hmtl_name
            elif code == 'W' and main_hmtl_emotion == '平静':
                 # 如果只有 W 编码
                 # 则 W (Aroused) 映射为焦虑 (高唤醒状态)
                 main_hmtl_emotion = '焦虑'
    
    # 2. 获取 HMTL 标签 (通过 label_mapper)
    hmtl_labels = get_hmtl_labels(main_hmtl_emotion)
    
    # 3. 计算 Arousal 和 Valence 值
    true_A = INTENSITY_MAP.get(max_intensity, 0.5) 
    
    polarity = hmtl_labels['label_3'] 
    intensity_val = INTENSITY_MAP.get(max_intensity, 0.5)
    
    if polarity == LABEL_3_MAP['积极']:
        true_V = intensity_val 
    elif polarity == LABEL_3_MAP['消极']:
        true_V = -intensity_val 
    else: # 平静
        true_V = 0.0 
        true_A = 0.2
    
    return {
        'label_4_emotion': hmtl_labels['label_4'],
        'label_3_polarity': hmtl_labels['label_3'],
        'true_arousal': round(true_A, 2),
        'true_valence': round(true_V, 2),
        'main_emotion_name': main_hmtl_emotion
    }


def generate_cnsce_labels():
    data = []
    if not os.path.exists(CNSCED_ROOT):
        print(f"警告: CNSCED_ROOT 路径 '{CNSCED_ROOT}' 不存在")
        return 0

    for root, _, files in os.walk(CNSCED_ROOT):
        for file in files:
            if file.endswith('.wav'):
                parsed_data = parse_filename(file)
                if parsed_data:
                    full_path = os.path.join(root, file).replace('\\', '/')
                    data.append({
                        'audio_full_path': full_path,
                        'label_4_emotion': parsed_data['label_4_emotion'],
                        'label_3_polarity': parsed_data['label_3_polarity'],
                        'true_arousal': parsed_data['true_arousal'],
                        'true_valence': parsed_data['true_valence']
                    })

    df = pd.DataFrame(data)
    if df.empty:
        print(f"警告: 在 '{CNSCED_ROOT}' 中未找到 CNSCED 音频文件")
        return 0
        
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"成功生成 {len(df)} 条 CNSCED 标签到 {OUTPUT_CSV}")

    print("\n--- 示例解析 (F0010-0015-S3_W3.wav) ---")
    example_data = parse_filename('F0010-0015-S3_W3.wav')
    print(f"主要情绪: {example_data['main_emotion_name']}")
    print(f"Arousal: {example_data['true_arousal']}")
    print(f"Valence: {example_data['true_valence']}")
    
    return len(df) # 

if __name__ == '__main__':
    generate_cnsce_labels()
