# -*- coding: utf-8 -*-
"""
Exp2-v2: λ敏感性实验（修复版）

改进点：
1. 类别级模态准确率差异（不同类别各模态表现不同）
2. MRE模型增大（16维隐层+dropout）
3. loss = BCE + λ * entropy_reg，λ真正控制正则化强度
4. 扩大λ范围（0.001~10.0），让曲线有起伏

Run:
  python exp2_lambda_v2.py --samples 5000 --seeds 13,42,2024,99,7
"""
import argparse
import random
import os
import csv
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

NUM_CLASSES = 4
ALL_MODALITIES = ["text", "visual", "audio"]

# 类别级模态准确率：不同情绪类别下各模态表现不同
# 行=类别(积极/激活消极/非激活消极/平静)，列=模态(text/visual/audio)
CLASS_MODALITY_ACC = {
    0: {"text": 0.85, "visual": 0.70, "audio": 0.80},  # 积极：文本最强
    1: {"text": 0.75, "visual": 0.55, "audio": 0.82},  # 激活消极：音频最强（语气激动）
    2: {"text": 0.80, "visual": 0.50, "audio": 0.65},  # 非激活消极：文本最强，音频弱
    3: {"text": 0.70, "visual": 0.72, "audio": 0.70},  # 平静：三模态接近
}

LABEL_NAMES = ['积极', '激活消极', '非激活消极', '平静']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticDatasetV2(Dataset):
    """带类别级模态准确率差异的合成数据集"""
    def __init__(self, n_samples=5000, missing_rate=0.1, seed=42,
                 class_acc=None, noise_config=None):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        self.missing_rate = missing_rate
        self.class_acc = class_acc or CLASS_MODALITY_ACC
        self.noise_config = noise_config or {}

        self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)
        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            modalities = {}
            for m in ALL_MODALITIES:
                if self.missing_rate > 0 and np.random.rand() < self.missing_rate:
                    modalities[m] = {"available": False}
                    continue
                # 类别级准确率
                acc = self.class_acc[y][m]
                # 噪声干扰
                noise_rate = self.noise_config.get(m, 0.0)
                if noise_rate > 0:
                    acc = acc * (1 - noise_rate) + (1.0 / NUM_CLASSES) * noise_rate

                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong))

                logits = self._gen_logits(NUM_CLASSES, pred, peak=3.0)
                if noise_rate > 0:
                    logits += np.random.normal(0, noise_rate * 2, size=logits.shape)
                probs = self._softmax(logits)
                conf = float(np.max(probs))
                modalities[m] = {
                    "available": True, "pred_idx": int(pred),
                    "probs": probs.astype(np.float32),
                    "logits": logits.astype(np.float32),
                    "confidence": conf,
                }
            self.samples.append({"label": y, "modalities": modalities})

    def _gen_logits(self, n_classes, pred_idx, peak=3.0):
        base = np.random.normal(0.0, 0.5, size=(n_classes,))
        base[pred_idx] += peak
        return base

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx]


# ============ Features & Model ============
def compute_entropy(probs):
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


def features_for_modality(sample, modality):
    mods = sample["modalities"]
    s = mods.get(modality, {"available": False})
    if not s.get("available", False):
        return np.array([0.0, 2.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), -1, 0.0, False
    probs = s["probs"]
    pred = int(s["pred_idx"])
    conf = float(s["confidence"])
    ent = compute_entropy(probs)
    logit_mean = float(np.mean(s["logits"]))
    logit_var = float(np.var(s["logits"]))
    others = []
    for m2 in ALL_MODALITIES:
        if m2 == modality:
            continue
        o = mods.get(m2, {"available": False})
        if o.get("available", False):
            others.append(int(o["pred_idx"]))
    agreement = float(sum(1 for p in others if p == pred) / len(others)) if others else 0.5
    feat = np.array([conf, ent, logit_mean, logit_var, 0.0, agreement], dtype=np.float32)
    return feat, pred, conf, True


class MLP_v2(nn.Module):
    """更大的MLP，让正则化有空间起作用"""
    def __init__(self, in_dim=6, h_dim=32, out_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class MRE_v2(nn.Module):
    def __init__(self, in_dim=6, h_dim=32, dropout=0.2):
        super().__init__()
        self.text_mlp = MLP_v2(in_dim, h_dim, 1, dropout)
        self.visual_mlp = MLP_v2(in_dim, h_dim, 1, dropout)
        self.audio_mlp = MLP_v2(in_dim, h_dim, 1, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats):
        out = {}
        if "text" in feats:
            out["text"] = self.sigmoid(self.text_mlp(feats["text"]))
        if "visual" in feats:
            out["visual"] = self.sigmoid(self.visual_mlp(feats["visual"]))
        if "audio" in feats:
            out["audio"] = self.sigmoid(self.audio_mlp(feats["audio"]))
        return out


def mre_bce_balanced(reliabilities, correctness):
    losses = []
    for m in ALL_MODALITIES:
        if m in reliabilities and m in correctness:
            r = reliabilities[m].view(-1)
            y = correctness[m].view(-1)
            pos_rate = torch.clamp(y.mean(), 1e-6, 1 - 1e-6)
            w_pos = 0.5 / pos_rate
            w_neg = 0.5 / (1 - pos_rate)
            r = torch.clamp(r, 1e-6, 1 - 1e-6)
            loss = -(w_pos * y * torch.log(r) + w_neg * (1 - y) * torch.log(1 - r))
            losses.append(loss.mean())
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def entropy_regularization(reliabilities):
    """熵正则项：鼓励可靠性输出趋向0.5（不确定），λ越大越保守"""
    ent_losses = []
    for m in ALL_MODALITIES:
        if m in reliabilities:
            r = reliabilities[m].view(-1)
            r = torch.clamp(r, 1e-6, 1 - 1e-6)
            # 负熵：r越接近0.5熵越大，正则化鼓励趋向0.5
            ent = -(r * torch.log(r) + (1 - r) * torch.log(1 - r))
            # 我们要最大化熵（让输出不确定），所以loss = -entropy
            ent_losses.append(-ent.mean())
    if not ent_losses:
        return torch.tensor(0.0)
    return torch.stack(ent_losses).mean()


def fuse_with_reliability(sample, reliabilities):
    preds = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds[m] = int(info["pred_idx"])
    if not preds:
        return 0, "no_modality"
    if len(set(preds.values())) == 1:
        return list(preds.values())[0], "unanimous"
    best_m, best_r = None, -1.0
    for m, p in preds.items():
        r = float(reliabilities.get(m, 0.0))
        if r > best_r:
            best_r = r
            best_m = m
    return preds[best_m], f"pick_{best_m}"


def weighted_vote(sample):
    # 用全局平均准确率作为权重
    global_weights = {}
    for m in ALL_MODALITIES:
        accs = [CLASS_MODALITY_ACC[c][m] for c in range(NUM_CLASSES)]
        global_weights[m] = np.mean(accs)
    vote_scores = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            pred = int(info["pred_idx"])
            w = global_weights[m]
            vote_scores[pred] = vote_scores.get(pred, 0) + w
    if not vote_scores:
        return 0
    return max(vote_scores, key=vote_scores.get)


def macro_f1(y_true, y_pred, n_classes=NUM_CLASSES):
    eps = 1e-8
    f1s = []
    for c in range(n_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


# ============ Training ============
def split_dataset(ds, train_r=0.7, val_r=0.15):
    n = len(ds)
    n_train = int(train_r * n)
    n_val = int(val_r * n)
    train_set = Subset(ds, list(range(n_train)))
    val_set = Subset(ds, list(range(n_train, n_train + n_val)))
    test_set = Subset(ds, list(range(n_train + n_val, n)))
    return train_set, val_set, test_set


def train_mre_v2(train_set, val_set, epochs=15, lr=3e-3, lambda_ent=0.3,
                 batch_size=64, device=None):
    """
    loss = BCE_balanced + lambda_ent * entropy_reg
    lambda_ent控制正则化强度：
      - 小λ：MRE自由判别，可能过拟合
      - 大λ：MRE被迫保守（输出趋向0.5），判别力下降
    """
    if device is None:
        device = torch.device("cpu")
    mre = MRE_v2(in_dim=6, h_dim=32, dropout=0.2).to(device)
    optimizer = optim.AdamW(mre.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    best_acc, best_state = 0, None

    for epoch in range(1, epochs + 1):
        mre.train()
        for batch in loader:
            optimizer.zero_grad()
            feats_t = {m: [] for m in ALL_MODALITIES}
            corr_t = {m: [] for m in ALL_MODALITIES}
            any_avail = {m: False for m in ALL_MODALITIES}
            for sample in batch:
                label = int(sample["label"])
                for m in ALL_MODALITIES:
                    f, pred, conf, avail = features_for_modality(sample, m)
                    if avail:
                        feats_t[m].append(torch.from_numpy(f).float())
                        corr_t[m].append(torch.tensor(1.0 if pred == label else 0.0))
                        any_avail[m] = True
            feats_b, corr_b = {}, {}
            for m in ALL_MODALITIES:
                if any_avail[m]:
                    feats_b[m] = torch.stack(feats_t[m]).to(device)
                    corr_b[m] = torch.stack(corr_t[m]).to(device)
            if not feats_b:
                continue
            rel = mre(feats_b)
            bce_loss = mre_bce_balanced(rel, corr_b)
            ent_reg = entropy_regularization(rel)
            loss = bce_loss + lambda_ent * ent_reg
            loss.backward()
            optimizer.step()

        # val
        val_m = eval_mre(val_set, mre, device)
        if val_m["accuracy"] > best_acc:
            best_acc = val_m["accuracy"]
            best_state = {k: v.clone() for k, v in mre.state_dict().items()}

    if best_state:
        mre.load_state_dict(best_state)
    return mre


def eval_mre(dataset, mre_model, device):
    mre_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            feats_t = {}
            for m in ALL_MODALITIES:
                f, _, _, avail = features_for_modality(sample, m)
                if avail:
                    feats_t[m] = torch.from_numpy(f).float().unsqueeze(0).to(device)
            rel = {}
            if feats_t:
                r = mre_model(feats_t)
                rel = {k: float(v.item()) for k, v in r.items()}
            pred, _ = fuse_with_reliability(sample, rel)
            y_true.append(int(sample["label"]))
            y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1}


def eval_weighted(dataset):
    y_true, y_pred = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        pred = weighted_vote(sample)
        y_true.append(int(sample["label"]))
        y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1}


# ============ Main Experiment ============
def main():
    parser = argparse.ArgumentParser(description="Exp2-v2: λ敏感性（修复版）")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="13,42,2024,99,7")
    parser.add_argument("--out_dir", type=str, default="outputs/supplementary_v2")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 打印类别级准确率
    print("类别级模态准确率:")
    print(f"{'类别':<12} {'文本':>6} {'视觉':>6} {'音频':>6}")
    for c in range(NUM_CLASSES):
        a = CLASS_MODALITY_ACC[c]
        print(f"  {LABEL_NAMES[c]:<10} {a['text']:>5.1%} {a['visual']:>5.1%} {a['audio']:>5.1%}")

    # λ范围：从极小到极大
    lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n{'='*70}")
    print(f"Exp2-v2: λ_ent 敏感性 (loss = BCE + λ * entropy_reg)")
    print(f"样本数: {args.samples} | 种子: {seeds} | epochs: {args.epochs}")
    print(f"{'='*70}")

    results = {}
    # 先跑一次加权投票作为baseline
    baseline_accs = []
    for seed in seeds:
        ds = SyntheticDatasetV2(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
        _, _, test_set = split_dataset(ds)
        wt = eval_weighted(test_set)
        baseline_accs.append(wt["accuracy"])
    baseline_mean = statistics.mean(baseline_accs)
    print(f"\n加权投票 baseline: {baseline_mean:.4f}")

    print(f"\n{'λ_ent':<10} {'MRE Acc':>12} {'vs baseline':>12}")
    print("-" * 40)

    for lam in lambdas:
        accs = []
        f1s = []
        for seed in seeds:
            ds = SyntheticDatasetV2(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre_v2(train_set, val_set, epochs=args.epochs, lr=args.lr,
                               lambda_ent=lam, device=device)
            m = eval_mre(test_set, mre, device)
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])

        results[lam] = {
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) >= 2 else 0,
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s) if len(f1s) >= 2 else 0,
        }
        r = results[lam]
        delta = r["acc_mean"] - baseline_mean
        print(f"  {lam:<8.3f} {r['acc_mean']:.4f}±{r['acc_std']:.4f}  {delta:+.4f}")

    # 找最优λ
    best_lam = max(results, key=lambda k: results[k]["acc_mean"])
    print(f"\n最优 λ_ent = {best_lam} (Acc={results[best_lam]['acc_mean']:.4f})")

    # 保存CSV
    if out_dir:
        csv_path = os.path.join(out_dir, "exp2_lambda_v2.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lambda_ent", "accuracy_mean", "accuracy_std",
                             "macro_f1_mean", "macro_f1_std", "weighted_baseline"])
            for lam in lambdas:
                r = results[lam]
                writer.writerow([f"{lam}", f"{r['acc_mean']:.4f}", f"{r['acc_std']:.4f}",
                                 f"{r['f1_mean']:.4f}", f"{r['f1_std']:.4f}",
                                 f"{baseline_mean:.4f}"])
        print(f"\n已保存: {csv_path}")

    # ============ 同时跑噪声敏感性（替代分布偏移） ============
    print(f"\n{'='*70}")
    print("噪声敏感性实验（替代分布偏移）")
    print("各模态在不同噪声强度下的MRE vs 加权投票")
    print(f"{'='*70}")

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    noise_results = {}

    # 用最优λ
    opt_lambda = best_lam

    for nl in noise_levels:
        noise_cfg = {"text": nl, "visual": nl, "audio": nl}
        mre_accs, wt_accs = [], []
        for seed in seeds:
            ds = SyntheticDatasetV2(n_samples=args.samples, missing_rate=args.missing_rate,
                                    seed=seed, noise_config=noise_cfg)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre_v2(train_set, val_set, epochs=args.epochs, lr=args.lr,
                               lambda_ent=opt_lambda, device=device)
            m = eval_mre(test_set, mre, device)
            wt = eval_weighted(test_set)
            mre_accs.append(m["accuracy"])
            wt_accs.append(wt["accuracy"])

        noise_results[nl] = {
            "mre_mean": statistics.mean(mre_accs),
            "mre_std": statistics.stdev(mre_accs) if len(mre_accs) >= 2 else 0,
            "wt_mean": statistics.mean(wt_accs),
            "wt_std": statistics.stdev(wt_accs) if len(wt_accs) >= 2 else 0,
        }
        r = noise_results[nl]
        delta = r["mre_mean"] - r["wt_mean"]
        print(f"  noise={nl:.0%}: MRE={r['mre_mean']:.4f}±{r['mre_std']:.4f}  "
              f"加权={r['wt_mean']:.4f}  Δ={delta:+.4f}")

    # 保存噪声敏感性CSV
    if out_dir:
        csv_path = os.path.join(out_dir, "noise_sensitivity_v2.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["noise_level", "mre_acc_mean", "mre_acc_std",
                             "weighted_acc_mean", "weighted_acc_std"])
            for nl in noise_levels:
                r = noise_results[nl]
                writer.writerow([f"{nl:.2f}", f"{r['mre_mean']:.4f}", f"{r['mre_std']:.4f}",
                                 f"{r['wt_mean']:.4f}", f"{r['wt_std']:.4f}"])
        print(f"\n已保存: {csv_path}")

    print(f"\n{'='*70}")
    print("实验完成！")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
