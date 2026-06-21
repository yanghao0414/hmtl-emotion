import torch
import os

MODEL_DIR = r'd:\bigcreate\06_模型文件'

# 检查视觉模型
print('=== 视觉模型 ===')
for name in ['visual_hmtl_v4_best.pt', 'visual_hmtl_v3_best.pt', 'visual_hmtl_v2_best.pt', 'visual_hmtl_best.pt']:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        continue
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f'\n{name} ({os.path.getsize(path)/1e6:.1f}MB):')
        if isinstance(ckpt, dict):
            print(f'  Keys: {list(ckpt.keys())}')
            if 'history' in ckpt:
                h = ckpt['history']
                if isinstance(h, dict):
                    for k in h:
                        vals = h[k]
                        if isinstance(vals, list) and len(vals) > 0:
                            print(f'  {k}: last={vals[-1]:.4f}')
            for key in ['best_acc', 'epoch', 'eval_metrics', 'best_metrics']:
                if key in ckpt:
                    print(f'  {key}: {ckpt[key]}')
    except Exception as e:
        print(f'  Error: {e}')

# 检查音频模型
print('\n\n=== 音频模型 ===')
for name in ['audio_hmtl_v2_best.pt', 'audio_best_hmtl.pt', 'audio_hmtl_trained.pt']:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        continue
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f'\n{name} ({os.path.getsize(path)/1e6:.1f}MB):')
        if isinstance(ckpt, dict):
            print(f'  Keys: {list(ckpt.keys())}')
            if 'history' in ckpt:
                h = ckpt['history']
                if isinstance(h, dict):
                    for k in h:
                        vals = h[k]
                        if isinstance(vals, list) and len(vals) > 0:
                            print(f'  {k}: last={vals[-1]:.4f}')
            for key in ['best_acc', 'epoch', 'eval_metrics', 'best_metrics']:
                if key in ckpt:
                    print(f'  {key}: {ckpt[key]}')
        else:
            print(f'  Type: {type(ckpt)}')
    except Exception as e:
        print(f'  Error: {e}')

# 检查根目录的模型
print('\n\n=== 根目录模型 ===')
for name in ['best_model_v2.pt', 'final_model_v2.pt', 'smp2020_final_model.pt']:
    path = os.path.join(r'd:\bigcreate', name)
    if not os.path.exists(path):
        continue
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        print(f'\n{name} ({os.path.getsize(path)/1e6:.1f}MB):')
        if isinstance(ckpt, dict):
            print(f'  Keys: {list(ckpt.keys())}')
            if 'history' in ckpt:
                h = ckpt['history']
                if isinstance(h, dict):
                    for k in h:
                        vals = h[k]
                        if isinstance(vals, list) and len(vals) > 0:
                            print(f'  {k}: last={vals[-1]:.4f}')
            for key in ['best_acc', 'epoch', 'eval_metrics', 'best_metrics']:
                if key in ckpt:
                    print(f'  {key}: {ckpt[key]}')
    except Exception as e:
        print(f'  Error: {e}')
