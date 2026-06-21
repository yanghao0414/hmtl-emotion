import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
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
    from audio_hmtl_classifier import AudioHMTLClassifier
    from cnsce_label_generator import generate_cnsce_labels
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# --- 配置路径 ---
CACHE_PATH = "05_数据集/audio_features_cache.pt"

# --- 1. 缓存版 Audio Dataset ---
class CachedAudioDataset(Dataset):
    def __init__(self, label_df, cached_features):
        """
        缓存版音频数据集
        
        Args:
            label_df: 标签 DataFrame，由 train_test_split 分割
            cached_features: 缓存特征字典，key 为 CSV 行索引
        """
        # 不做 reset_index
        self.label_df = label_df
        self.cached_features = cached_features
        
        # 筛选有效索引
        # label_df.index 对应 CSV 原始行号
        self.valid_indices = []
        for orig_idx in self.label_df.index:
            if orig_idx in self.cached_features:
                self.valid_indices.append(orig_idx)
        
        print(f"  - DataFrame 总数: {len(self.label_df)}")
        print(f"  - 有效样本数: {len(self.valid_indices)}")
        
        if len(self.valid_indices) == 0:
            print("警告: 无有效样本")
            print(f"    DataFrame 索引范围: {self.label_df.index.min()} - {self.label_df.index.max()}")
            print(f"    缓存 key 示例: {list(self.cached_features.keys())[:5]}")
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 获取原始索引
        original_idx = self.valid_indices[idx]
        row = self.label_df.loc[original_idx]
        
        # 获取缓存特征
        features = self.cached_features[original_idx]
        input_values = features['input_values']
        attention_mask = features['attention_mask']

        # 获取标签
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
    """自定义 Collate 函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None 

    # 动态Padding
    max_len = max(item['input_values'].shape[0] for item in batch)
    
    padded_batch = {}
    for key in batch[0].keys():
        if key in ['input_values', 'attention_mask']:
            # Padding 处理
            padded_tensors = [
                torch.nn.functional.pad(item[key], (0, max_len - item[key].shape[0]), 'constant', 0.0) 
                for item in batch
            ]
            padded_batch[key] = torch.stack(padded_tensors)
        else:
            # 直接堆叠
            padded_batch[key] = torch.stack([item[key] for item in batch])
            
    return padded_batch


# --- 2. 损失函数 + 类别权重 ---
# 根据训练集分布计算的权重
# 类别 0: 1160 条, 类别 1: 240 条, 类别 2: 87 条, 类别 3: 2441 条
WEIGHTS_TENSOR = torch.tensor([0.8466, 4.0917, 11.2874, 0.4023], dtype=torch.float)

def HMTL_Audio_Loss(
    logits_4, logits_3, pred_A, pred_V, 
    labels_4, labels_3, true_A, true_V, 
    lambda_1=0.2, lambda_A=0.01, lambda_V=0.01  # 降低回归损失权重
):
    # L4 使用加权损失
    ce_loss_weighted = nn.CrossEntropyLoss(weight=WEIGHTS_TENSOR.to(labels_4.device))
    ce_loss_unweighted = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss() 
    
    L_4_emotion = ce_loss_weighted(logits_4, labels_4)  # 加权交叉熵
    L_3_polarity = ce_loss_unweighted(logits_3, labels_3)
    L_arousal = mse_loss(pred_A.squeeze(-1), true_A)
    L_valence = mse_loss(pred_V.squeeze(-1), true_V)
    
    L_total = (
        L_4_emotion
        + lambda_1 * L_3_polarity
        + lambda_A * L_arousal 
        + lambda_V * L_valence
    )
    
    return L_total, L_4_emotion, L_3_polarity, L_arousal, L_valence


# --- 3. 评估函数 ---
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
            
            preds_4 = torch.argmax(logits_4, dim=1).cpu().numpy()
            all_preds_4.extend(preds_4)
            all_labels_4.extend(labels_4.cpu().numpy())
            
            all_preds_A.extend(pred_A.squeeze(-1).cpu().numpy())
            all_labels_A.extend(true_A.cpu().numpy())

    if len(all_labels_4) == 0:
        model.train()
        return 0.0, 0.0

    acc_4 = accuracy_score(all_labels_4, all_preds_4)
    mse_A = mean_squared_error(all_labels_A, all_preds_A)
    
    model.train()
    return acc_4, mse_A


# --- 4. 数据加载 ---
def load_cached_data(csv_path="05_数据集/audio_hmtl_labels.csv", cache_path=CACHE_PATH, test_frac=0.2):
    """加载缓存数据"""
    
    # 检查缓存文件
    if not os.path.exists(cache_path):
        print(f"缓存文件不存在: {cache_path}")
        print("请先运行: python modules/audio_hmtl/preprocess_features.py")
        return None, None
    
    # 加载标签
    try:
        label_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"标签文件不存在: {csv_path}")
        return None, None
    
    print(f"加载了 {len(label_df)} 条标签")
    
    # 加载缓存
    print(f"加载特征缓存...")
    cached_features = torch.load(cache_path)
    cache_size = os.path.getsize(cache_path) / (1024**3)
    print(f"缓存包含 {len(cached_features)} 条特征 ({cache_size:.2f} GB)")
    
    # 分割数据集
    train_df, test_df = train_test_split(label_df, test_size=test_frac, random_state=42)
    
    print(f"\n创建数据集...")
    print(f"训练集:")
    train_dataset = CachedAudioDataset(train_df, cached_features)
    print(f"测试集:")
    test_dataset = CachedAudioDataset(test_df, cached_features)
    
    return train_dataset, test_dataset


# --- 5. 训练函数 ---
def train_model(num_epochs=10, batch_size=16, learning_rate=5e-5, model_save_path="06_模型文件/audio_best_hmtl.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    # 1. 加载数据
    train_dataset, test_dataset = load_cached_data()
    if train_dataset is None:
        return

    # 2. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Windows 下用 0
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # 3. 初始化模型
    print(f"\n初始化模型...")
    model = AudioHMTLClassifier().to(device)
    
    # --- 差异化学习率 (分层微调) ---
    high_lr = learning_rate * 10  # 分类头 10 倍学习率
    
    # 分类头参数
    head_params = [
        *model.dim_reducer.parameters(), 
        *model.classifier_4.parameters(),
        *model.classifier_3.parameters(), 
        *model.regressor_A.parameters(), 
        *model.regressor_V.parameters()
    ]
    
    # 骨干网络参数
    base_params = list(model.wav2vec2.parameters())
    
    optimizer_grouped_parameters = [
        {'params': base_params, 'lr': learning_rate},      # Wav2Vec2骨干: 5e-05
        {'params': head_params, 'lr': high_lr}             # 分类头: 5e-04
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    print(f"  → 骨干学习率: {learning_rate}")
    print(f"  → 分类头学习率: {high_lr} (10x)")
    # --- 差异化学习率结束 ---

    best_acc_4 = 0.0
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 4. 训练循环
    print(f"\n开始训练 {num_epochs} 个 epochs...")
    print("=" * 60)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            if batch is None: continue 
            
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

            # 计算损失 + 加权
            L_total, L_4, L_3, L_A, L_V = HMTL_Audio_Loss(
                logits_4, logits_3, pred_A, pred_V, 
                labels_4, labels_3, true_A, true_V
            )
            
            # 反向传播
            L_total.backward()
            optimizer.step()
            
            total_loss += L_total.item()
            
            # 进度条
            pbar.set_postfix({
                'Loss': f'{L_total.item():.2e}', 
                'L4': f'{L_4.item():.2e}',
                'LA': f'{L_A.item():.2e}',
            })
            
        avg_loss = total_loss / len(train_loader)
        
        # 5. 评估
        acc_4, mse_A = evaluate_model(model, test_loader, device)
        
        print(f"\nEpoch {epoch+1} 结果:")
        print(f"  → 平均损失: {avg_loss:.4e}")
        print(f"  → 测试集 4类准确率: {acc_4:.4f}")
        print(f"  → 测试集 Arousal MSE: {mse_A:.4f}")

        if acc_4 > best_acc_4:
            best_acc_4 = acc_4
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型 (4类准确率: {best_acc_4:.4f})")

    print("\n" + "=" * 60)
    print(f"训练完成! 最佳准确率: {best_acc_4:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    print("=" * 60)
    print("音频 Wav2Vec 2.0 HMTL 训练 (缓存版)")
    print("=" * 60)
    
    # 开始训练
    train_model(
        num_epochs=3,   # CPU 下先跑 3 epochs
        batch_size=8,   # 较小的 batch size 防止内存溢出
        learning_rate=1e-4  # 较大学习率
    )
