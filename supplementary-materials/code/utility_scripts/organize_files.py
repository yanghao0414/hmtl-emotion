#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动整理HMTL项目文件
创建清晰的文件夹结构
"""

import os
import shutil
from pathlib import Path

def organize_hmtl_project():
    """整理HMTL项目文件"""
    base_dir = Path(r"d:\bigcreate")
    
    print("="*60)
    print("HMTL项目文件整理")
    print("="*60)
    
    # 创建文件夹结构
    folders = {
        '01_文档': base_dir / '01_文档',
        '02_模型代码': base_dir / '02_模型代码',
        '03_训练脚本': base_dir / '03_训练脚本',
        '04_评估工具': base_dir / '04_评估工具',
        '05_数据文件': base_dir / '05_数据文件',
        '06_模型文件': base_dir / '06_模型文件',
        '07_示例代码': base_dir / '07_示例代码',
        '08_工具脚本': base_dir / '08_工具脚本'
    }
    
    # 创建文件夹
    for folder_name, folder_path in folders.items():
        folder_path.mkdir(exist_ok=True)
        print(f"✓ 创建文件夹: {folder_name}")
    
    # 文件分类规则
    file_mappings = {
        # 文档
        '01_文档': [
            'HMTL快速上手.md',
            'HMTL使用指南.md',
            'HMTL情绪分类方案分析.md',
            'HMTL优化策略实施方案.md',
            'HMTL_V2优化执行总结.md',
            '情绪分类标准与数据分布报告.md',
            '数据平衡标注计划_快速参考.md'
        ],
        # 模型代码
        '02_模型代码': [
            'hmtl_model.py',
            'hmtl_model_v2.py',
            'hmtl_utils.py',
            'hmtl_dataset.py'
        ],
        # 训练脚本
        '03_训练脚本': [
            'hmtl_train.py',
            'hmtl_train_v2.py',
            'convert_to_hmtl.py'
        ],
        # 评估工具
        '04_评估工具': [
            'hmtl_evaluate.py',
            'verify_dataloader.py',
            'verify_hmtl_mapping.py'
        ],
        # 数据文件
        '05_数据文件': [
            'training_set_hmtl.json',
            'training_set_hmtl_augmented.json',
            'eval_set_hmtl.json',
            'eval_set.json',
            'auto_labeled_candidates.json',
            'reviewed_samples.json'
        ],
        # 示例代码
        '07_示例代码': [
            'simple_example.py',
            'quick_use_hmtl.py',
            'hmtl_api.py',
            'show_candidates.py'
        ],
        # 工具脚本
        '08_工具脚本': [
            'data_augmentation.py',
            'auto_label_with_rules.py',
            'review_tool.py',
            'organize_files.py'
        ]
    }
    
    # 移动文件
    print("\n移动文件...")
    moved_count = 0
    
    for target_folder, files in file_mappings.items():
        target_path = folders[target_folder]
        
        for filename in files:
            src = base_dir / filename
            dst = target_path / filename
            
            if src.exists() and src != dst:
                try:
                    shutil.copy2(src, dst)
                    print(f"  ✓ {filename} → {target_folder}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ {filename}: {e}")
    
    # 移动模型文件夹
    model_folders = ['hmtl_models', 'hmtl_models_v2']
    for model_folder in model_folders:
        src = base_dir / model_folder
        if src.exists():
            dst = folders['06_模型文件'] / model_folder
            if src != dst:
                try:
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    print(f"  ✓ {model_folder}/ → 06_模型文件/")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ {model_folder}: {e}")
    
    print(f"\n✅ 文件整理完成！共整理 {moved_count} 个文件/文件夹")
    
    # 创建README
    create_readme(folders['01_文档'])
    
    return folders

def create_readme(doc_folder):
    """创建项目README"""
    readme_content = """# HMTL情绪识别项目 - 文件结构说明

## 📁 文件夹结构

```
d:\\bigcreate\\
├── 01_文档/                    # 所有文档和报告
├── 02_模型代码/                 # 核心模型定义
├── 03_训练脚本/                 # 训练相关脚本
├── 04_评估工具/                 # 评估和验证工具
├── 05_数据文件/                 # 训练和评估数据
├── 06_模型文件/                 # 训练好的模型文件
│   ├── hmtl_models/            # V1模型
│   └── hmtl_models_v2/         # V2模型（优化版）
├── 07_示例代码/                 # 快速开始示例
└── 08_工具脚本/                 # 辅助工具
```

## 🚀 快速开始

### 1. 使用V2模型（推荐）

```python
from sys import path
path.append(r'd:\\bigcreate\\02_模型代码')
path.append(r'd:\\bigcreate\\04_评估工具')

from hmtl_evaluate import HMTLPredictor

predictor = HMTLPredictor(
    model_path=r"d:\\bigcreate\\06_模型文件\\hmtl_models_v2\\best_model_v2.pt"
)

result = predictor.predict("我很开心")
print(result)
```

### 2. 查看文档

- **快速上手**: `01_文档/HMTL快速上手.md`
- **使用指南**: `01_文档/HMTL使用指南.md`
- **优化策略**: `01_文档/HMTL优化策略实施方案.md`

### 3. 运行示例

```bash
cd d:\\bigcreate\\07_示例代码
python simple_example.py
```

## 📊 模型版本对比

| 版本 | 准确率 | 特点 | 推荐 |
|-----|-------|-----|-----|
| V1 | 79.0% | 基础版本 | ⭐⭐⭐ |
| V2 | 83-85%预期 | Focal Loss + Arousal优化 | ⭐⭐⭐⭐⭐ |

## 📞 需要帮助？

查看 `01_文档/` 中的详细文档。
"""
    
    readme_path = doc_folder / 'README_文件结构.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✓ 创建README: {readme_path}")

if __name__ == "__main__":
    organize_hmtl_project()
    print("\n" + "="*60)
    print("文件整理完成！项目结构已优化。")
    print("="*60)
