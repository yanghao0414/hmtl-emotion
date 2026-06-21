import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys

# 导入标签映射工具
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
try:
    from label_mapper import get_hmtl_labels 
except ImportError:
    pass 

LABEL_CSV_PATH = "05_数据集/audio_hmtl_labels.csv"

def get_class_weights(df):
    # 计算类别权重
    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
    
    # 统计 4 类 (label_4_emotion) 分布
    class_counts = train_df['label_4_emotion'].value_counts().sort_index()
    total_samples = class_counts.sum()
    num_classes = len(class_counts)
    
    # 逆频率平滑 (Inverse Frequency Smoothing)
    # 权重 = 总样本数 / (类别数 * 该类样本数)
    class_weights = total_samples / (num_classes * class_counts.values)
    
    # 转换为 PyTorch Tensor
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # 格式化输出
    weights_str = ', '.join([f'{w:.4f}' for w in weights_tensor.tolist()])
    
    print("="*60)
    print(f"类别权重分析 4类 (训练集):")
    print("-" * 60)
    for i, count in class_counts.items():
        if i < len(weights_tensor):
            print(f"  类别 {i}: 数量 {count}, 权重 {weights_tensor[i]:.4f}")
    print("="*60)
    print(f"权重向量: [{weights_str}]")
    
    return weights_str

if __name__ == '__main__':
    try:
        if not os.path.exists(LABEL_CSV_PATH):
             print(f"找不到文件 {LABEL_CSV_PATH}")
             sys.exit(1)
        df = pd.read_csv(LABEL_CSV_PATH)
        # 计算并保存权重
        weights_str = get_class_weights(df)
        with open("class_weights.txt", "w") as f:
            f.write(weights_str)
        print("权重已保存到 class_weights.txt")
        
    except Exception as e:
        print(f"错误: {e}")
