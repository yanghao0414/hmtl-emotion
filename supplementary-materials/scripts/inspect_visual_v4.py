"""检查V4模型的state_dict结构，反推模型架构"""
import torch

ckpt = torch.load(r'd:\bigcreate\06_模型文件\visual_hmtl_v4_best.pt', map_location='cpu', weights_only=False)

if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
else:
    sd = ckpt

# 只打印非backbone的key和shape
print("=== 非backbone层 ===")
for k, v in sd.items():
    if not k.startswith('backbone.features'):
        print(f"  {k}: {v.shape}")

print("\n=== backbone最后几层 ===")
for k, v in sd.items():
    if k.startswith('backbone.features.8') or k.startswith('backbone.classifier'):
        print(f"  {k}: {v.shape}")
