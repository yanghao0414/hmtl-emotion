#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试工具
简单快速地测试你的多模态融合系统
"""

import sys
import os
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "path_bootstrap.py").exists():
        _p_str = str(_p)
        if _p_str not in sys.path:
            sys.path.insert(0, _p_str)
        break

from path_bootstrap import bootstrap

bootstrap()

from multimodal_fusion_system import MultimodalFusionSystem

def quick_test():
    """快速测试函数"""
    print("🚀 快速测试多模态融合系统")
    print("="*50)
    
    # 初始化系统
    fusion_system = MultimodalFusionSystem()
    if not fusion_system.initialize_models():
        print("❌ 系统初始化失败")
        return
    
    # 测试用例
    test_cases = [
        "我今天心情很好，工作进展顺利",
        "我很担心明天的考试",
        "感觉很失落",
        "心情很平静"
    ]
    
    print(f"\n📊 测试 {len(test_cases)} 个用例...")
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n【测试 {i}】{text}")
        print("-" * 40)
        
        try:
            # 比较所有融合策略
            fusion_system.compare_fusion_strategies(text)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print(f"\n✅ 快速测试完成！")

if __name__ == "__main__":
    quick_test()
