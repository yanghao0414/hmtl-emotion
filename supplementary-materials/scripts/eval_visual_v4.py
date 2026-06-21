"""
评估视觉V4模型在测试集上的性能
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# ============ 模型定义 (V4架构，从checkpoint反推) ============
class VisualHMTLClassifierV4(nn.Module):
    """
    V4架构：shared_fc有BatchNorm，classifier_4有BatchNorm，
    regressor_A/V有BatchNorm+更深层
    """
    def __init__(self, dropout=0.3, pretrained=False):
        super().__init__()
        self.backbone = models.efficientnet_b2(weights=None)
        feature_dim = self.backbone.classifier[1].in_features  # 1408
        self.backbone.classifier = nn.Identity()
        
        # shared_fc: Dropout -> Linear(1408,512) -> ReLU -> BatchNorm(512)
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # classifier_7: Linear(512,256) -> ReLU -> Dropout -> Linear(256,7)
        self.classifier_7 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 7)
        )
        
        # classifier_4: Linear(512,256) -> ReLU -> BatchNorm(256) -> Dropout -> Linear(256,4)
        self.classifier_4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 4)
        )
        
        # classifier_3: Linear(512,128) -> ReLU -> Dropout -> Linear(128,3)
        self.classifier_3 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 3)
        )
        
        # regressor_A: Linear(512,256) -> ReLU -> BatchNorm(256) -> Linear(256,64) -> ReLU -> Linear(64,1)
        self.regressor_A = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # regressor_V: Linear(512,256) -> ReLU -> BatchNorm(256) -> Linear(256,64) -> ReLU -> Linear(64,1)
        self.regressor_V = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)
        return {
            'label_7_logits': self.classifier_7(shared),
            'label_4_logits': self.classifier_4(shared),
            'label_3_logits': self.classifier_3(shared),
            'arousal': self.regressor_A(shared).squeeze(-1),
            'valence': self.regressor_V(shared).squeeze(-1)
        }

# ============ 数据集 ============
class VisualEvalDataset(Dataset):
    def __init__(self, csv_path, image_root, split='Test', transform=None):
        df = pd.read_csv(csv_path)
        # 只取Test集
        df = df[df['image_path'].str.contains(f'\\\\{split}\\\\|/{split}/', regex=True, case=False)]
        
        self.samples = []
        for _, row in df.iterrows():
            # 修正路径: 从CSV路径提取相对路径部分
            orig_path = row['image_path']
            # 提取 "archive (3)\Test\anger/image0000006.jpg" 部分
            rel_parts = orig_path.split('visual_data_temp\\')
            if len(rel_parts) > 1:
                rel_path = rel_parts[1]
            else:
                rel_path = orig_path
            
            img_path = os.path.join(image_root, rel_path)
            if os.path.exists(img_path):
                self.samples.append({
                    'image_path': img_path,
                    'label_7': int(row['label_7']),
                    'label_4': int(row['label_4']),
                    'label_3': int(row['label_3']),
                    'arousal': float(row['arousal']),
                    'valence': float(row['valence'])
                })
        
        print(f"加载 {len(self.samples)} 个{split}样本 (共 {len(df)} 条记录)")
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s['image_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label_7': s['label_7'],
            'label_4': s['label_4'],
            'label_3': s['label_3'],
            'arousal': s['arousal'],
            'valence': s['valence']
        }

# ============ 评估 ============
def evaluate():
    MODEL_PATH = r'd:\bigcreate\06_模型文件\visual_hmtl_v4_best.pt'
    CSV_PATH = r'd:\bigcreate\05_数据文件\labels\visual_hmtl_labels.csv'
    IMAGE_ROOT = r'd:\bigcreate\05_数据文件\visual_data_temp'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    print("加载视觉V4模型...")
    model = VisualHMTLClassifierV4(pretrained=False)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # 可能是纯state_dict
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()
    print("模型加载成功")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    dataset = VisualEvalDataset(CSV_PATH, IMAGE_ROOT, split='Test', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 评估
    all_pred_4, all_true_4 = [], []
    all_pred_7, all_true_7 = [], []
    all_pred_3, all_true_3 = [], []
    all_mae_a, all_mae_v = [], []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            outputs = model(images)
            
            pred_4 = outputs['label_4_logits'].argmax(dim=1).cpu().numpy()
            pred_7 = outputs['label_7_logits'].argmax(dim=1).cpu().numpy()
            pred_3 = outputs['label_3_logits'].argmax(dim=1).cpu().numpy()
            
            all_pred_4.extend(pred_4)
            all_pred_7.extend(pred_7)
            all_pred_3.extend(pred_3)
            all_true_4.extend(batch['label_4'].numpy())
            all_true_7.extend(batch['label_7'].numpy())
            all_true_3.extend(batch['label_3'].numpy())
            
            mae_a = torch.abs(outputs['arousal'].cpu() - batch['arousal']).numpy()
            mae_v = torch.abs(outputs['valence'].cpu() - batch['valence']).numpy()
            all_mae_a.extend(mae_a)
            all_mae_v.extend(mae_v)
    
    # 打印结果
    labels_4 = ['积极', '激活消极', '非激活消极', '平静']
    labels_7 = ['愤怒', '焦虑', '快乐', '悲伤', '失望', '支持', '平静']
    labels_3 = ['积极', '消极', '平静']
    
    print("\n" + "="*60)
    print("视觉V4模型评估结果")
    print("="*60)
    
    acc_4 = np.mean(np.array(all_pred_4) == np.array(all_true_4))
    acc_7 = np.mean(np.array(all_pred_7) == np.array(all_true_7))
    acc_3 = np.mean(np.array(all_pred_3) == np.array(all_true_3))
    
    print(f"\n4类准确率: {acc_4:.4f} ({acc_4*100:.2f}%)")
    print(f"7类准确率: {acc_7:.4f} ({acc_7*100:.2f}%)")
    print(f"3类准确率: {acc_3:.4f} ({acc_3*100:.2f}%)")
    print(f"Arousal MAE: {np.mean(all_mae_a):.4f}")
    print(f"Valence MAE: {np.mean(all_mae_v):.4f}")
    
    print(f"\n--- 4类分类报告 ---")
    print(classification_report(all_true_4, all_pred_4, target_names=labels_4, zero_division=0))
    
    print(f"\n--- 7类分类报告 ---")
    print(classification_report(all_true_7, all_pred_7, target_names=labels_7, zero_division=0))
    
    print(f"\n--- 4类混淆矩阵 ---")
    cm4 = confusion_matrix(all_true_4, all_pred_4)
    print(f"{'':>12} {'积极':>8} {'激活消极':>8} {'非激活消极':>10} {'平静':>8}")
    for i, row in enumerate(cm4):
        print(f"{labels_4[i]:>12} {row[0]:>8} {row[1]:>8} {row[2]:>10} {row[3]:>8}")
    
    print(f"\n总样本数: {len(all_true_4)}")

if __name__ == '__main__':
    evaluate()
