import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 导入依赖
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from label_mapper import get_hmtl_labels 
    from audio_preprocessor import AudioPreprocessor 
    from audio_hmtl_classifier import AudioHMTLClassifier
    from cnsce_label_generator import generate_cnsce_labels
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)


# --- 1. Audio Dataset 数据集 ---

class AudioDataset(Dataset):
    def __init__(self, label_df):
        self.label_df = label_df
        self.preprocessor = AudioPreprocessor() 

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        audio_full_path = row['audio_full_path']
        
        processed_features = self.preprocessor.preprocess_audio(audio_full_path)
        
        if processed_features is None:
             # 返回 None，由 collate_fn 过滤
             return None
             
        input_values = processed_features['input_values']
        attention_mask = processed_features['attention_mask']

        labels_4 = torch.tensor(row['label_4_emotion'], dtype=torch.long)
        labels_3 = torch.tensor(row['label_3_polarity'], dtype=torch.long)
        true_A = torch.tensor(row['true_arousal'], dtype=torch.float)
        true_V = torch.tensor(row['true_valence'], dtype=torch.float)

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels_4': labels_4,
            'labels_3': labels_3,
            'true_A': true_A,
            'true_V': true_V
        }
        
def custom_collate_fn(batch):
    """自定义 Collate 函数，过滤 None 样本"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # 返回 None，由 train_model 跳过

    # Wav2Vec 2.0 需要 Padding
    # 对 input_values 进行 Padding
    max_len = max(item['input_values'].shape[0] for item in batch)
    
    padded_batch = {}
    for key in batch[0].keys():
        if key in ['input_values', 'attention_mask']:
            # 对 input_values 和 attention_mask 进行 Padding
            if key == 'input_values':
                 # 用 0 填充 padding
                padded_tensors = [
                    torch.nn.functional.pad(item[key], (0, max_len - item[key].shape[0]), 'constant', 0.0) 
                    for item in batch
                ]
            else: # attention_mask, 用 0 填充 padding
                padded_tensors = [
                    torch.nn.functional.pad(item[key], (0, max_len - item[key].shape[0]), 'constant', 0.0) 
                    for item in batch
                ]
            padded_batch[key] = torch.stack(padded_tensors)
        else:
            # 直接堆叠
            padded_batch[key] = torch.stack([item[key] for item in batch])
            
    return padded_batch


# --- 2. 数据加载 ---

def load_audio_data(csv_path="05_数据集/audio_hmtl_labels.csv", test_frac=0.2):
    try:
        label_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 {csv_path}，请先运行 generate_cnsce_labels()")
        return None, None
        
    print(f"加载了 {len(label_df)} 条标签")

    train_df, test_df = train_test_split(label_df, test_size=test_frac, random_state=42)

    train_dataset = AudioDataset(train_df.reset_index(drop=True))
    test_dataset = AudioDataset(test_df.reset_index(drop=True))
    
    return train_dataset, test_dataset


# --- 3. HMTL 损失函数 ---

def HMTL_Audio_Loss(
    logits_4, logits_3, pred_A, pred_V, 
    labels_4, labels_3, true_A, true_V, 
    lambda_1=0.2, lambda_A=1.5, lambda_V=0.5 
):
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss() 
    
    L_4_emotion = ce_loss(logits_4, labels_4)
    L_3_polarity = ce_loss(logits_3, labels_3)
    L_arousal = mse_loss(pred_A.squeeze(-1), true_A)
    L_valence = mse_loss(pred_V.squeeze(-1), true_V)
    
    L_total = (
        L_4_emotion
        + lambda_1 * L_3_polarity
        + lambda_A * L_arousal 
        + lambda_V * L_valence
    )
    
    return L_total, L_4_emotion, L_3_polarity, L_arousal, L_valence

# --- 4. 评估函数 ---

def evaluate_model(model, data_loader, device):
    model.eval()
    
    all_preds_4, all_labels_4 = [], []
    all_preds_A, all_labels_A = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch is None: continue 
            
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_4 = batch['labels_4'].to(device)
            true_A = batch['true_A'].to(device)

            logits_4, _, pred_A, _ = model(input_values, attention_mask)
            
            # 4类预测
            preds_4 = torch.argmax(logits_4, dim=1).cpu().numpy()
            all_preds_4.extend(preds_4)
            all_labels_4.extend(labels_4.cpu().numpy())
            
            # Arousal 预测
            all_preds_A.extend(pred_A.squeeze(-1).cpu().numpy())
            all_labels_A.extend(true_A.cpu().numpy())

    # 空数据检查
    if len(all_labels_4) == 0:
        print("警告: 评估数据为空")
        model.train()
        return 0.0, 0.0
    
    # 计算指标
    acc_4 = accuracy_score(all_labels_4, all_preds_4)
    mse_A = mean_squared_error(all_labels_A, all_preds_A)
    
    model.train()
    
    return acc_4, mse_A

# --- 5. 训练函数 ---
def train_model(num_epochs=10, batch_size=8, learning_rate=5e-5, model_save_path="06_模型文件/audio_best_hmtl.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载数据
    train_dataset, test_dataset = load_audio_data()
    if train_dataset is None: return

    # 使用 collate_fn 过滤 None 样本
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 2. 初始化模型
    model = AudioHMTLClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_acc_4 = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # 创建目录

    # 3. 训练循环
    print(f"开始 Wav2Vec 2.0 HMTL 训练 ({len(train_dataset)} 条数据)...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            if batch is None: continue # 跳过空批次
            
            optimizer.zero_grad()
            
            # 转移到设备
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_4 = batch['labels_4'].to(device)
            labels_3 = batch['labels_3'].to(device)
            true_A = batch['true_A'].to(device)
            true_V = batch['true_V'].to(device)

            # 前向传播
            logits_4, logits_3, pred_A, pred_V = model(input_values, attention_mask)

            # 计算 HMTL 损失
            L_total, L_4, L_3, L_A, L_V = HMTL_Audio_Loss(
                logits_4, logits_3, pred_A, pred_V, 
                labels_4, labels_3, true_A, true_V
            )
            
            # 反向传播
            L_total.backward()
            optimizer.step()
            
            total_loss += L_total.item()
            
            # 进度条: 显示损失
            pbar.set_postfix({
                'Loss': f'{L_total.item():.2e}', 
                'L4': f'{L_4.item():.2e}',
                'L3': f'{L_3.item():.2e}',
                'LA': f'{L_A.item():.2e}',
            })
            
        avg_loss = total_loss / len(train_loader)
        
        # 4. 评估
        acc_4, mse_A = evaluate_model(model, test_loader, device)
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  → 平均损失: {avg_loss:.4e} (越小越好)")
        print(f"  → 测试集 4类准确率: {acc_4:.4f}")
        print(f"  → 测试集 Arousal MSE: {mse_A:.4f}")

        if acc_4 > best_acc_4:
            best_acc_4 = acc_4
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型: {model_save_path} (4类准确率 {best_acc_4:.4f})")

    print("训练完成! HMTL 音频模型已保存")

if __name__ == '__main__':
    print("--- Wav2Vec 2.0 HMTL 音频训练 ---")
    
    # 步骤 1: 生成标签
    print("生成标签文件...")
    generate_cnsce_labels() 
    
    # 步骤 2: 训练模型
    train_model()
