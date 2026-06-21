import torch
import pandas as pd
import os
from transformers import Wav2Vec2Processor
import librosa
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 配置路径
LABEL_CSV_PATH = "05_数据集/audio_hmtl_labels.csv"
OUTPUT_CACHE_PATH = "05_数据集/audio_features_cache.pt"
PROCESSOR_NAME = "facebook/wav2vec2-base" 

def generate_cache():
    if not os.path.exists(LABEL_CSV_PATH):
        print(f"文件不存在: 请先生成标签: {LABEL_CSV_PATH}")
        return

    print(f"加载标签文件...")
    df = pd.read_csv(LABEL_CSV_PATH)
    
    print(f"加载 Wav2Vec2 处理器...")
    processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
    sampling_rate = processor.feature_extractor.sampling_rate
    
    cached_features = {}
    failed_files = []
    
    print(f"\n开始处理 {len(df)} 条音频...")
    print("=" * 60)
    
    # 使用 tqdm 显示进度
    pbar = tqdm(df.iterrows(), total=len(df), desc="处理音频")
    
    for index, row in pbar:
        audio_full_path = row['audio_full_path']
        
        if not os.path.exists(audio_full_path):
            failed_files.append(f"文件不存在: {audio_full_path}")
            continue
            
        try:
            # 1. librosa 加载音频
            speech, rate = librosa.load(audio_full_path, sr=sampling_rate)
            
            # 2. Wav2Vec 2.0 特征提取
            processed = processor(
                speech, 
                sampling_rate=sampling_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            # 3. 保存特征 (后续 Batch 处理)
            input_values = processed.input_values.squeeze(0)
            attention_mask = torch.ones(input_values.shape, dtype=torch.long)
            
            # 缓存特征
            cached_features[index] = {
                'input_values': input_values,
                'attention_mask': attention_mask
            }
            
            # 更新进度条
            pbar.set_postfix({
                '成功': len(cached_features),
                '失败': len(failed_files)
            })
            
        except Exception as e:
            failed_files.append(f"{os.path.basename(audio_full_path)}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"处理结果:")
    print(f"  成功缓存: {len(cached_features)} 条")
    print(f"  失败数量: {len(failed_files)} 条")
    
    if failed_files:
        print(f"\n失败文件示例 (前5个):")
        for fail in failed_files[:5]:
            print(f"  - {fail}")

    # 4. 保存缓存
    print(f"\n保存缓存到 {OUTPUT_CACHE_PATH}...")
    torch.save(cached_features, OUTPUT_CACHE_PATH)
    
    # 统计信息
    file_size = os.path.getsize(OUTPUT_CACHE_PATH) / (1024**3)  # GB
    print(f"缓存完成")
    print(f"  - 样本数: {len(cached_features)}")
    print(f"  - 文件大小: {file_size:.2f} GB")
    print(f"  - 保存路径: {OUTPUT_CACHE_PATH}")

if __name__ == '__main__':
    print("=" * 60)
    print("音频特征预处理工具")
    print("=" * 60)
    generate_cache()
    print("\n处理完成")
