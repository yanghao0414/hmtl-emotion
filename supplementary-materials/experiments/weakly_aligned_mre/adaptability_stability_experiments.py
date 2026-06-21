# -*- coding: utf-8 -*-
"""
适配性 + 抗风险性 + 稳定性 综合实验

Exp4: 噪声干扰实验 — 文本mask/音频噪声/视觉模糊，不同强度vs性能
Exp5: 小样本学习曲线 — 10%/20%/50%/100%训练比例下融合性能
Exp6: 分布偏移实验 — 改变测试集情绪分布
Exp7: 多种子稳定性 — 10次种子，报告Mean±Std
Exp8: 置信度稳定性 — 噪声下MRE置信度分布变化

Run:
  python adaptability_stability_experiments.py --samples 5000 --seeds 13,42,2024 --exp all
"""
import argparse
import random
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import statistics

NUM_CLASSES = 4
ALL_MODALITIES = ["text", "visual", "audio"]

REAL_ACCURACIES = {
    "text":   0.789,
    "visual": 0.629,
    "audio":  0.754,
}

LABEL_NAMES = ['积极', '激活消极', '非激活消极', '平静']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============ Dataset ============
class SyntheticDataset(Dataset):
    def __init__(self, n_samples=5000, modality_acc=None, missing_rate=0.1, seed=42,
                 noise_config=None, class_distribution=None):
        """
        noise_config: dict, e.g. {"text": 0.2, "visual": 0.1, "audio": 0.15}
            对每个模态，以noise_rate概率将预测翻转为随机错误类
        class_distribution: list of 4 floats, 各类别比例，默认均匀
        """
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        if modality_acc is None:
            modality_acc = REAL_ACCURACIES.copy()
        self.modality_acc = modality_acc
        self.missing_rate = missing_rate
        self.noise_config = noise_config or {}

        # 生成标签（支持自定义分布）
        if class_distribution is not None:
            probs = np.array(class_distribution, dtype=np.float64)
            probs = probs / probs.sum()
            self.labels = np.random.choice(NUM_CLASSES, size=(self.n,), p=probs).astype(np.int64)
        else:
            self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)

        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            modalities = {}
            for m in ALL_MODALITIES:
                if self.missing_rate > 0 and np.random.rand() < self.missing_rate:
                    modalities[m] = {"available": False}
                    continue
                acc = self.modality_acc[m]
                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong))

                # 噪声干扰：以noise_rate概率翻转预测
                noise_rate = self.noise_config.get(m, 0.0)
                if noise_rate > 0 and np.random.rand() < noise_rate:
                    wrong = [c for c in range(NUM_CLASSES) if c != pred]
                    pred = int(np.random.choice(wrong))

                logits = self._gen_logits(NUM_CLASSES, pred, peak=3.0)
                # 噪声也影响logits的清晰度
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


# ============ MRE Model (same as supplementary) ============
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


class MLP(nn.Module):
    def __init__(self, in_dim=6, h_dim=8, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
    def forward(self, x):
        return self.net(x)


class MRE(nn.Module):
    def __init__(self, in_dim=6, h_dim=8):
        super().__init__()
        self.text_mlp = MLP(in_dim, h_dim, 1)
        self.visual_mlp = MLP(in_dim, h_dim, 1)
        self.audio_mlp = MLP(in_dim, h_dim, 1)
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


def weighted_vote(sample, weights=None):
    if weights is None:
        weights = REAL_ACCURACIES.copy()
    vote_scores = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            pred = int(info["pred_idx"])
            w = weights.get(m, 0.5)
            vote_scores[pred] = vote_scores.get(pred, 0) + w
    if not vote_scores:
        return 0
    return max(vote_scores, key=vote_scores.get)


def majority_vote(sample):
    from collections import Counter
    preds = []
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds.append(int(info["pred_idx"]))
    if not preds:
        return 0
    counter = Counter(preds)
    return counter.most_common(1)[0][0]


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


# ============ Training & Evaluation ============
def split_dataset(ds, train_r=0.7, val_r=0.15):
    n = len(ds)
    n_train = int(train_r * n)
    n_val = int(val_r * n)
    train_set = Subset(ds, list(range(n_train)))
    val_set = Subset(ds, list(range(n_train, n_train + n_val)))
    test_set = Subset(ds, list(range(n_train + n_val, n)))
    return train_set, val_set, test_set


def train_mre(train_set, val_set, epochs=10, lr=3e-3, lambda_mre=0.3, batch_size=64, device=None):
    if device is None:
        device = torch.device("cpu")
    mre = MRE(in_dim=6, h_dim=8).to(device)
    optimizer = optim.AdamW(mre.parameters(), lr=lr)
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
            loss = lambda_mre * mre_bce_balanced(rel, corr_b)
            loss.backward()
            optimizer.step()
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


def eval_mre_with_confidence(dataset, mre_model, device):
    """评估并返回每个样本的MRE置信度"""
    mre_model.eval()
    records = []
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
            pred, path = fuse_with_reliability(sample, rel)
            label = int(sample["label"])
            records.append({
                "label": label, "pred": int(pred),
                "correct": int(pred == label),
                "rel_text": rel.get("text", -1),
                "rel_visual": rel.get("visual", -1),
                "rel_audio": rel.get("audio", -1),
            })
    return records


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


def eval_majority(dataset):
    y_true, y_pred = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        pred = majority_vote(sample)
        y_true.append(int(sample["label"]))
        y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1}


# ============================================================
# Exp4: 噪声干扰实验
# ============================================================
def exp4_noise_robustness(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp4: 噪声干扰实验")
    print("=" * 70)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    results = {}

    # 4a: 单模态噪声（只给一个模态加噪声，其他不变）
    print("\n--- 4a: 单模态噪声 ---")
    for target_m in ALL_MODALITIES:
        name = {"text": "文本", "visual": "视觉", "audio": "音频"}[target_m]
        print(f"\n  对 {name} 加噪声:")
        results[target_m] = {}
        for nl in noise_levels:
            noise_cfg = {target_m: nl}
            seed_mre, seed_wt, seed_mj = [], [], []
            for seed in seeds:
                ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                      seed=seed, noise_config=noise_cfg)
                train_set, val_set, test_set = split_dataset(ds)
                mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                               lambda_mre=args.lambda_mre, device=device)
                mre_m = eval_mre(test_set, mre, device)
                wt_m = eval_weighted(test_set)
                mj_m = eval_majority(test_set)
                seed_mre.append(mre_m["accuracy"])
                seed_wt.append(wt_m["accuracy"])
                seed_mj.append(mj_m["accuracy"])
            results[target_m][nl] = {
                "mre": {"mean": statistics.mean(seed_mre), "std": statistics.stdev(seed_mre) if len(seed_mre) >= 2 else 0},
                "weighted": {"mean": statistics.mean(seed_wt), "std": statistics.stdev(seed_wt) if len(seed_wt) >= 2 else 0},
                "majority": {"mean": statistics.mean(seed_mj), "std": statistics.stdev(seed_mj) if len(seed_mj) >= 2 else 0},
            }
            r = results[target_m][nl]
            print(f"    noise={nl:.0%}: MRE={r['mre']['mean']:.4f}±{r['mre']['std']:.4f}  "
                  f"加权={r['weighted']['mean']:.4f}  多数={r['majority']['mean']:.4f}")

    # 4b: 全模态同时加噪声
    print("\n--- 4b: 全模态同时加噪声 ---")
    results["all"] = {}
    for nl in noise_levels:
        noise_cfg = {"text": nl, "visual": nl, "audio": nl}
        seed_mre, seed_wt, seed_mj = [], [], []
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                  seed=seed, noise_config=noise_cfg)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=args.lambda_mre, device=device)
            mre_m = eval_mre(test_set, mre, device)
            wt_m = eval_weighted(test_set)
            mj_m = eval_majority(test_set)
            seed_mre.append(mre_m["accuracy"])
            seed_wt.append(wt_m["accuracy"])
            seed_mj.append(mj_m["accuracy"])
        results["all"][nl] = {
            "mre": {"mean": statistics.mean(seed_mre), "std": statistics.stdev(seed_mre) if len(seed_mre) >= 2 else 0},
            "weighted": {"mean": statistics.mean(seed_wt), "std": statistics.stdev(seed_wt) if len(seed_wt) >= 2 else 0},
            "majority": {"mean": statistics.mean(seed_mj), "std": statistics.stdev(seed_mj) if len(seed_mj) >= 2 else 0},
        }
        r = results["all"][nl]
        print(f"  noise={nl:.0%}: MRE={r['mre']['mean']:.4f}±{r['mre']['std']:.4f}  "
              f"加权={r['weighted']['mean']:.4f}  多数={r['majority']['mean']:.4f}")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp4_noise_robustness.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_modality", "noise_level", "mre_acc_mean", "mre_acc_std",
                             "weighted_acc_mean", "majority_acc_mean"])
            for target_m in list(ALL_MODALITIES) + ["all"]:
                for nl in noise_levels:
                    r = results[target_m][nl]
                    writer.writerow([target_m, f"{nl:.2f}",
                                     f"{r['mre']['mean']:.4f}", f"{r['mre']['std']:.4f}",
                                     f"{r['weighted']['mean']:.4f}", f"{r['majority']['mean']:.4f}"])
        print(f"\n  已保存: {csv_path}")

    return results


# ============================================================
# Exp5: 小样本学习曲线
# ============================================================
def exp5_sample_efficiency(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp5: 小样本学习曲线")
    print("=" * 70)

    train_ratios = [0.05, 0.10, 0.20, 0.50, 0.70, 1.00]
    results = {}

    for ratio in train_ratios:
        seed_mre, seed_wt, seed_mj = [], [], []
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
            train_set, val_set, test_set = split_dataset(ds)

            # 从训练集中抽取子集
            n_train = len(train_set)
            k = max(1, int(n_train * ratio))
            random.seed(seed)
            indices = random.sample(range(n_train), k)
            sub_train = Subset(train_set, indices)

            mre = train_mre(sub_train, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=args.lambda_mre, device=device)
            mre_m = eval_mre(test_set, mre, device)
            wt_m = eval_weighted(test_set)
            mj_m = eval_majority(test_set)
            seed_mre.append(mre_m["accuracy"])
            seed_wt.append(wt_m["accuracy"])
            seed_mj.append(mj_m["accuracy"])

        results[ratio] = {
            "mre": {"mean": statistics.mean(seed_mre), "std": statistics.stdev(seed_mre) if len(seed_mre) >= 2 else 0},
            "weighted": {"mean": statistics.mean(seed_wt), "std": statistics.stdev(seed_wt) if len(seed_wt) >= 2 else 0},
            "majority": {"mean": statistics.mean(seed_mj), "std": statistics.stdev(seed_mj) if len(seed_mj) >= 2 else 0},
            "n_train_samples": k,
        }
        r = results[ratio]
        print(f"  {ratio:>5.0%} ({k:>4d}样本): MRE={r['mre']['mean']:.4f}±{r['mre']['std']:.4f}  "
              f"加权={r['weighted']['mean']:.4f}  多数={r['majority']['mean']:.4f}")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp5_sample_efficiency.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["train_ratio", "n_samples", "mre_acc_mean", "mre_acc_std",
                             "weighted_acc_mean", "majority_acc_mean"])
            for ratio in train_ratios:
                r = results[ratio]
                writer.writerow([f"{ratio:.2f}", r["n_train_samples"],
                                 f"{r['mre']['mean']:.4f}", f"{r['mre']['std']:.4f}",
                                 f"{r['weighted']['mean']:.4f}", f"{r['majority']['mean']:.4f}"])
        print(f"\n  已保存: {csv_path}")

    return results


# ============================================================
# Exp6: 分布偏移实验
# ============================================================
def exp6_distribution_shift(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp6: 分布偏移实验")
    print("=" * 70)

    # 训练集用均匀分布，测试集用不同偏移分布
    shift_configs = {
        "均匀分布(基线)":     [0.25, 0.25, 0.25, 0.25],
        "积极主导(60%)":      [0.60, 0.15, 0.15, 0.10],
        "消极主导(60%)":      [0.10, 0.35, 0.35, 0.20],
        "极端不平衡(80%积极)": [0.80, 0.07, 0.07, 0.06],
        "只有两类":           [0.50, 0.50, 0.00, 0.00],
        "长尾分布":           [0.50, 0.25, 0.15, 0.10],
    }

    results = {}
    for shift_name, dist in shift_configs.items():
        # 处理0概率的情况
        dist_safe = [max(d, 0.001) for d in dist]

        seed_mre, seed_wt, seed_mj = [], [], []
        for seed in seeds:
            # 训练集：均匀分布
            ds_train_full = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                             seed=seed, class_distribution=[0.25, 0.25, 0.25, 0.25])
            train_set, val_set, _ = split_dataset(ds_train_full)

            # 测试集：偏移分布
            ds_test = SyntheticDataset(n_samples=int(args.samples * 0.15), missing_rate=args.missing_rate,
                                       seed=seed + 10000, class_distribution=dist_safe)

            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=args.lambda_mre, device=device)
            mre_m = eval_mre(ds_test, mre, device)
            wt_m = eval_weighted(ds_test)
            mj_m = eval_majority(ds_test)
            seed_mre.append(mre_m["accuracy"])
            seed_wt.append(wt_m["accuracy"])
            seed_mj.append(mj_m["accuracy"])

        results[shift_name] = {
            "distribution": dist,
            "mre": {"mean": statistics.mean(seed_mre), "std": statistics.stdev(seed_mre) if len(seed_mre) >= 2 else 0},
            "weighted": {"mean": statistics.mean(seed_wt), "std": statistics.stdev(seed_wt) if len(seed_wt) >= 2 else 0},
            "majority": {"mean": statistics.mean(seed_mj), "std": statistics.stdev(seed_mj) if len(seed_mj) >= 2 else 0},
        }
        r = results[shift_name]
        print(f"  {shift_name:<20s}: MRE={r['mre']['mean']:.4f}±{r['mre']['std']:.4f}  "
              f"加权={r['weighted']['mean']:.4f}  多数={r['majority']['mean']:.4f}")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp6_distribution_shift.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["shift_name", "distribution", "mre_acc_mean", "mre_acc_std",
                             "weighted_acc_mean", "majority_acc_mean"])
            for name, r in results.items():
                writer.writerow([name, str(r["distribution"]),
                                 f"{r['mre']['mean']:.4f}", f"{r['mre']['std']:.4f}",
                                 f"{r['weighted']['mean']:.4f}", f"{r['majority']['mean']:.4f}"])
        print(f"\n  已保存: {csv_path}")

    return results


# ============================================================
# Exp7: 多种子稳定性 (10次)
# ============================================================
def exp7_multi_seed_stability(args, seeds_10, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp7: 多种子稳定性 (10次)")
    print("=" * 70)

    all_mre, all_wt, all_mj = [], [], []
    for seed in seeds_10:
        ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
        train_set, val_set, test_set = split_dataset(ds)
        mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                       lambda_mre=args.lambda_mre, device=device)
        mre_m = eval_mre(test_set, mre, device)
        wt_m = eval_weighted(test_set)
        mj_m = eval_majority(test_set)
        all_mre.append(mre_m["accuracy"])
        all_wt.append(wt_m["accuracy"])
        all_mj.append(mj_m["accuracy"])
        print(f"  Seed {seed:>5d}: MRE={mre_m['accuracy']:.4f}  加权={wt_m['accuracy']:.4f}  多数={mj_m['accuracy']:.4f}")

    results = {
        "mre": {"mean": statistics.mean(all_mre), "std": statistics.stdev(all_mre),
                "min": min(all_mre), "max": max(all_mre), "values": all_mre},
        "weighted": {"mean": statistics.mean(all_wt), "std": statistics.stdev(all_wt),
                     "min": min(all_wt), "max": max(all_wt), "values": all_wt},
        "majority": {"mean": statistics.mean(all_mj), "std": statistics.stdev(all_mj),
                     "min": min(all_mj), "max": max(all_mj), "values": all_mj},
    }

    print(f"\n  汇总 (n={len(seeds_10)}):")
    for method, name in [("mre", "MRE"), ("weighted", "加权投票"), ("majority", "多数投票")]:
        r = results[method]
        print(f"    {name:<8s}: {r['mean']:.4f}±{r['std']:.4f}  [{r['min']:.4f}, {r['max']:.4f}]")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp7_multi_seed.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "mre_acc", "weighted_acc", "majority_acc"])
            for i, seed in enumerate(seeds_10):
                writer.writerow([seed, f"{all_mre[i]:.4f}", f"{all_wt[i]:.4f}", f"{all_mj[i]:.4f}"])
            writer.writerow(["MEAN", f"{results['mre']['mean']:.4f}",
                             f"{results['weighted']['mean']:.4f}", f"{results['majority']['mean']:.4f}"])
            writer.writerow(["STD", f"{results['mre']['std']:.4f}",
                             f"{results['weighted']['std']:.4f}", f"{results['majority']['std']:.4f}"])
        print(f"\n  已保存: {csv_path}")

    return results


# ============================================================
# Exp8: 置信度稳定性（噪声下MRE置信度分布变化）
# ============================================================
def exp8_confidence_stability(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp8: 置信度稳定性（噪声下MRE可靠性分布变化）")
    print("=" * 70)

    noise_levels = [0.0, 0.10, 0.20, 0.30, 0.50]
    results = {}

    for nl in noise_levels:
        noise_cfg = {"text": nl, "visual": nl, "audio": nl}
        all_records = []
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                  seed=seed, noise_config=noise_cfg)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=args.lambda_mre, device=device)
            records = eval_mre_with_confidence(test_set, mre, device)
            all_records.extend(records)

        # 统计
        acc = sum(r["correct"] for r in all_records) / len(all_records)
        rel_stats = {}
        for m in ALL_MODALITIES:
            key = f"rel_{m}"
            vals = [r[key] for r in all_records if r[key] >= 0]
            if vals:
                rel_stats[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                }
        results[nl] = {"accuracy": acc, "reliability": rel_stats, "n_samples": len(all_records)}

        name_map = {"text": "文本", "visual": "视觉", "audio": "音频"}
        rel_str = "  ".join(f"{name_map[m]}={rel_stats[m]['mean']:.3f}±{rel_stats[m]['std']:.3f}"
                            for m in ALL_MODALITIES if m in rel_stats)
        print(f"  noise={nl:.0%}: Acc={acc:.4f}  可靠性: {rel_str}")

    # 计算置信度变化率
    if 0.0 in results and 0.5 in results:
        print("\n  --- 置信度变化率 (noise=0% → 50%) ---")
        for m in ALL_MODALITIES:
            name = {"text": "文本", "visual": "视觉", "audio": "音频"}[m]
            r0 = results[0.0]["reliability"].get(m, {})
            r5 = results[0.5]["reliability"].get(m, {})
            if r0 and r5:
                mean_change = r5["mean"] - r0["mean"]
                std_change = r5["std"] - r0["std"]
                print(f"    {name}: mean变化={mean_change:+.4f}, std变化={std_change:+.4f}")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp8_confidence_stability.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["noise_level", "accuracy",
                             "rel_text_mean", "rel_text_std",
                             "rel_visual_mean", "rel_visual_std",
                             "rel_audio_mean", "rel_audio_std"])
            for nl in noise_levels:
                r = results[nl]
                row = [f"{nl:.2f}", f"{r['accuracy']:.4f}"]
                for m in ALL_MODALITIES:
                    rs = r["reliability"].get(m, {})
                    row.extend([f"{rs.get('mean', 0):.4f}", f"{rs.get('std', 0):.4f}"])
                writer.writerow(row)
        print(f"\n  已保存: {csv_path}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="适配性+抗风险性+稳定性实验")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="13,42,2024")
    parser.add_argument("--out_dir", type=str, default="outputs/adaptability_stability")
    parser.add_argument("--exp", type=str, default="all",
                        help="运行哪个实验: 4, 5, 6, 7, 8, 或 all")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 10个种子用于稳定性实验
    seeds_10 = [7, 13, 42, 99, 123, 456, 789, 2024, 3141, 9999]

    print(f"适配性+抗风险性+稳定性实验")
    print(f"样本数: {args.samples} | 种子: {seeds} | 设备: {device}")
    print(f"真实模型准确率: T={REAL_ACCURACIES['text']:.1%} V={REAL_ACCURACIES['visual']:.1%} A={REAL_ACCURACIES['audio']:.1%}")

    run_all = args.exp == "all"
    all_results = {}

    if run_all or args.exp == "4":
        all_results["exp4"] = exp4_noise_robustness(args, seeds, device, out_dir)

    if run_all or args.exp == "5":
        all_results["exp5"] = exp5_sample_efficiency(args, seeds, device, out_dir)

    if run_all or args.exp == "6":
        all_results["exp6"] = exp6_distribution_shift(args, seeds, device, out_dir)

    if run_all or args.exp == "7":
        all_results["exp7"] = exp7_multi_seed_stability(args, seeds_10, device, out_dir)

    if run_all or args.exp == "8":
        all_results["exp8"] = exp8_confidence_stability(args, seeds, device, out_dir)

    print("\n" + "=" * 70)
    print("所有实验完成！")
    if out_dir:
        print(f"结果保存在: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
