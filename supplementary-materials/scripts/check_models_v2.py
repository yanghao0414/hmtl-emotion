import torch
import os

MODEL_DIR = r'd:\bigcreate\06_模型文件'

# 检查视觉模型
print('=== 视觉模型 ===')
for name in ['visual_hmtl_v4_best.pt', 'visual_hmtl_v3_best.pt', 'visual_hmtl_v2_best.pt', 'visual_hmtl_best.pt', 'visual_hmtl_trained.pt']:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f'\n{name} ({os.path.getsize(path)/1e6:.1f}MB):')
    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        # 只打印非state_dict的key
        meta_keys = [k for k in keys if 'state_dict' not in k]
        print(f'  Meta keys: {meta_keys}')
        for key in ['best_acc', 'epoch', 'eval_metrics', 'best_metrics', 'accuracy_4', 'accuracy_7', 'acc_4', 'acc_7']:
            if key in ckpt:
                print(f'  {key}: {ckpt[key]}')
        if 'history' in ckpt:
            h = ckpt['history']
            if isinstance(h, dict):
                for k in h:
                    vals = h[k]
                    if isinstance(vals, list) and len(vals) > 0:
                        print(f'  {k}: last={vals[-1]:.4f} (epochs={len(vals)})')
    else:
        # 可能是纯state_dict
        print(f'  Type: {type(ckpt).__name__}, keys count: {len(ckpt) if hasattr(ckpt, "__len__") else "N/A"}')

# 检查音频模型
print('\n\n=== 音频模型 ===')
for name in ['audio_hmtl_v2_best.pt', 'audio_best_hmtl.pt', 'audio_hmtl_trained.pt']:
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        continue
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f'\n{name} ({os.path.getsize(path)/1e6:.1f}MB):')
    if isinstance(ckpt, dict):
        meta_keys = [k for k in ckpt.keys() if 'state_dict' not in k and not k.startswith('wav2vec') and not k.startswith('dim_') and not k.startswith('classifier') and not k.startswith('regressor')]
        print(f'  Meta keys: {meta_keys}')
        for key in ['best_acc', 'epoch', 'eval_metrics', 'best_metrics', 'accuracy_4', 'accuracy_7', 'acc_4', 'acc_7']:
            if key in ckpt:
                print(f'  {key}: {ckpt[key]}')
        if 'history' in ckpt:
            h = ckpt['history']
            if isinstance(h, dict):
                for k in h:
                    vals = h[k]
                    if isinstance(vals, list) and len(vals) > 0:
                        print(f'  {k}: last={vals[-1]:.4f} (epochs={len(vals)})')
        # 检查是否是纯state_dict（没有model_state_dict key）
        has_model_key = 'model_state_dict' in ckpt
        if not has_model_key:
            # 可能整个dict就是state_dict
            sample_keys = [k for k in list(ckpt.keys())[:3]]
            print(f'  (looks like raw state_dict, sample keys: {sample_keys})')
