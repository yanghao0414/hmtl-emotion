#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 Visual HMTL 数据集
从 AffectNet 数据生成 HMTL 格式的标签文件
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # 02_模型代码/

from modules.visual_hmtl.affectnet_label_mapper import process_affectnet_dataset

# 配置路径
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
DATA_DIR = str(_PROJECT_ROOT / "05_数据文件" / "visual_data_temp" / "archive (3)")
LABELS_CSV = str(_PROJECT_ROOT / "05_数据文件" / "visual_data_temp" / "labels_full.csv")
OUTPUT_CSV = str(_PROJECT_ROOT / "05_数据文件" / "visual_hmtl_labels.csv")


def main():
    print("="*60)
    print("Visual HMTL 数据集准备")
    print("="*60)
    
    # 检查文件
    if not os.path.exists(LABELS_CSV):
        print(f"❌ 找不到标签文件: {LABELS_CSV}")
        return
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到数据目录: {DATA_DIR}")
        return
    
    # 处理数据集
    df = process_affectnet_dataset(
        data_dir=DATA_DIR,
        labels_csv=LABELS_CSV,
        output_csv=OUTPUT_CSV
    )
    
    print("\n" + "="*60)
    print("✅ 数据集准备完成！")
    print("="*60)
    print(f"总样本数: {len(df)}")
    print(f"标签文件: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
