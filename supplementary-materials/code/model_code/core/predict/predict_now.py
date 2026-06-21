#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL情绪预测 - 快速使用脚本
"""

import sys
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

from multimodal_fusion_v2 import MultimodalFusionSystemV2


def main():
    print("="*60)
    print("🎯 HMTL多模态情绪识别系统")
    print("="*60)
    
    # 初始化系统 (仅加载文本模型，速度快)
    print("\n🔄 正在加载模型...")
    system = MultimodalFusionSystemV2()
    system.initialize_models(load_visual=False, load_audio=False)
    
    print("\n✅ 系统就绪！")
    print("-"*60)
    print("输入文本进行情绪预测")
    print("输入 'quit' 或 'q' 退出")
    print("-"*60)
    
    while True:
        try:
            text = input("\n📝 请输入文本: ").strip()
            
            if text.lower() in ['quit', 'q', 'exit', '退出']:
                print("\n👋 再见！")
                break
            
            if not text:
                print("⚠️ 请输入有效文本")
                continue
            
            # 预测
            result = system.fuse(text=text)
            
            if result and 'fusion_result' in result:
                r = result['fusion_result']
                print("\n" + "="*40)
                print("📊 预测结果:")
                print(f"   4类分类: {r['4类分类']}")
                print(f"   7类情绪: {r['7类情绪']}")
                print(f"   3类极性: {r['3类极性']}")
                print(f"   唤醒度:  {r['arousal']:.3f}")
                print(f"   效价:    {r['valence']:.3f}")
                print("="*40)
            else:
                print("❌ 预测失败")
                
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
