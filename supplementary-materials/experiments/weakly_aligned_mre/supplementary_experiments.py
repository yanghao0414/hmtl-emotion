# -*- coding: utf-8 -*-
"""
补充实验：
  Exp1: 不同缺失率下的鲁棒性对比 (0%, 10%, 20%, 30%, 50%)
  Exp2: 超参数敏感性分析 (lambda_mre, lr)
  Exp3: 冲突样本分析 + 各模态可靠性分布

Run:
  python supplementary_experiments.py --samples 5000 --seeds 13,42,2024
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
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, n_samples=5000, modality_acc=None, missing_rate=0.1, seed=42):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        if modality_acc is None:
            modality_acc = REAL_ACCURACIES.copy()
        self.modality_acc = modality_acc
        self.missing_rate = missing_rate
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
                logits = self._gen_logits(NUM_CLASSES, pred, peak=3.0)
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


# ============ MRE Model ============
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


def eval_mre_detailed(dataset, mre_model, device):
    """详细评估：返回每个样本的可靠性、冲突状态等"""
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
            # 冲突检测
            preds_map = {}
            for m in ALL_MODALITIES:
                info = sample["modalities"].get(m, {"available": False})
                if info.get("available", False):
                    preds_map[m] = int(info["pred_idx"])
            is_conflict = len(set(preds_map.values())) > 1 if len(preds_map) >= 2 else False
            records.append({
                "idx": i, "label": label, "pred": int(pred),
                "correct": int(pred == label),
                "conflict": int(is_conflict),
                "path": path,
                "rel_text": rel.get("text", -1),
                "rel_visual": rel.get("visual", -1),
                "rel_audio": rel.get("audio", -1),
                "pred_text": preds_map.get("text", -1),
                "pred_visual": preds_map.get("visual", -1),
                "pred_audio": preds_map.get("audio", -1),
                "n_available": len(preds_map),
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


def eval_confidence(dataset):
    y_true, y_pred = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        rel = {}
        for m in ALL_MODALITIES:
            info = sample["modalities"].get(m, {"available": False})
            if info.get("available", False):
                rel[m] = float(info["confidence"])
        pred, _ = fuse_with_reliability(sample, rel)
        y_true.append(int(sample["label"]))
        y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1}


def split_dataset(ds, train_r=0.7, val_r=0.15):
    n = len(ds)
    n_train = int(train_r * n)
    n_val = int(val_r * n)
    train_set = torch.utils.data.Subset(ds, list(range(n_train)))
    val_set = torch.utils.data.Subset(ds, list(range(n_train, n_train + n_val)))
    test_set = torch.utils.data.Subset(ds, list(range(n_train + n_val, n)))
    return train_set, val_set, test_set


# ============ Exp1: 缺失率鲁棒性 ============
def exp1_missing_rate(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp1: 缺失率鲁棒性实验")
    print("=" * 70)

    missing_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = {}

    for mr in missing_rates:
        seed_results = {"mre": [], "weighted": [], "confidence": []}
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=mr, seed=seed)
            train_set, val_set, test_set = split_dataset(ds)
            # MRE
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=args.lambda_mre, device=device)
            mre_m = eval_mre(test_set, mre, device)
            seed_results["mre"].append(mre_m)
            # Weighted
            w_m = eval_weighted(test_set)
            seed_results["weighted"].append(w_m)
            # Confidence
            c_m = eval_confidence(test_set)
            seed_results["confidence"].append(c_m)

        results[mr] = {}
        for method in ["mre", "weighted", "confidence"]:
            accs = [r["accuracy"] for r in seed_results[method]]
            f1s = [r["macro_f1"] for r in seed_results[method]]
            results[mr][method] = {
                "accuracy": {"mean": statistics.mean(accs), "std": statistics.stdev(accs) if len(accs) >= 2 else 0},
                "macro_f1": {"mean": statistics.mean(f1s), "std": statistics.stdev(f1s) if len(f1s) >= 2 else 0}
            }

    # 打印
    print(f"\n{'缺失率':<10} {'MRE Acc':>12} {'加权 Acc':>12} {'置信度 Acc':>12} {'MRE优势':>10}")
    print("-" * 60)
    for mr in missing_rates:
        r = results[mr]
        mre_acc = r["mre"]["accuracy"]["mean"]
        w_acc = r["weighted"]["accuracy"]["mean"]
        c_acc = r["confidence"]["accuracy"]["mean"]
        delta = mre_acc - max(w_acc, c_acc)
        print(f"  {mr:<8.0%} {mre_acc:.4f}±{r['mre']['accuracy']['std']:.4f}"
              f"  {w_acc:.4f}±{r['weighted']['accuracy']['std']:.4f}"
              f"  {c_acc:.4f}±{r['confidence']['accuracy']['std']:.4f}"
              f"  {delta:+.4f}")

    # 保存CSV
    if out_dir:
        csv_path = os.path.join(out_dir, "exp1_missing_rate.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["missing_rate", "method", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"])
            for mr in missing_rates:
                for method in ["mre", "weighted", "confidence"]:
                    r = results[mr][method]
                    writer.writerow([f"{mr:.2f}", method,
                                     f"{r['accuracy']['mean']:.4f}", f"{r['accuracy']['std']:.4f}",
                                     f"{r['macro_f1']['mean']:.4f}", f"{r['macro_f1']['std']:.4f}"])
    return results


# ============ Exp2: 超参数敏感性 ============
def exp2_hyperparams(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp2: 超参数敏感性分析")
    print("=" * 70)

    # 2a: lambda_mre
    lambdas = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    lambda_results = {}
    print("\n--- lambda_mre 敏感性 ---")
    for lam in lambdas:
        accs = []
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_mre=lam, device=device)
            m = eval_mre(test_set, mre, device)
            accs.append(m["accuracy"])
        lambda_results[lam] = {
            "mean": statistics.mean(accs),
            "std": statistics.stdev(accs) if len(accs) >= 2 else 0
        }
        print(f"  λ={lam:<6.2f} Acc={lambda_results[lam]['mean']:.4f}±{lambda_results[lam]['std']:.4f}")

    # 2b: learning rate
    lrs = [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2]
    lr_results = {}
    print("\n--- learning rate 敏感性 ---")
    for lr_val in lrs:
        accs = []
        for seed in seeds:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=lr_val,
                           lambda_mre=args.lambda_mre, device=device)
            m = eval_mre(test_set, mre, device)
            accs.append(m["accuracy"])
        lr_results[lr_val] = {
            "mean": statistics.mean(accs),
            "std": statistics.stdev(accs) if len(accs) >= 2 else 0
        }
        print(f"  lr={lr_val:<8.1e} Acc={lr_results[lr_val]['mean']:.4f}±{lr_results[lr_val]['std']:.4f}")

    # 保存
    if out_dir:
        csv_path = os.path.join(out_dir, "exp2_lambda_sensitivity.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lambda_mre", "accuracy_mean", "accuracy_std"])
            for lam in lambdas:
                r = lambda_results[lam]
                writer.writerow([f"{lam:.2f}", f"{r['mean']:.4f}", f"{r['std']:.4f}"])

        csv_path = os.path.join(out_dir, "exp2_lr_sensitivity.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["learning_rate", "accuracy_mean", "accuracy_std"])
            for lr_val in lrs:
                r = lr_results[lr_val]
                writer.writerow([f"{lr_val:.1e}", f"{r['mean']:.4f}", f"{r['std']:.4f}"])

    return {"lambda": lambda_results, "lr": lr_results}


# ============ Exp3: 冲突样本分析 ============
def exp3_conflict_analysis(args, seeds, device, out_dir):
    print("\n" + "=" * 70)
    print("Exp3: 冲突样本分析 + 模态可靠性分布")
    print("=" * 70)

    all_records = []
    for seed in seeds:
        ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=seed)
        train_set, val_set, test_set = split_dataset(ds)
        mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                       lambda_mre=args.lambda_mre, device=device)
        records = eval_mre_detailed(test_set, mre, device)
        all_records.extend(records)

    # 分析
    total = len(all_records)
    conflict_records = [r for r in all_records if r["conflict"] == 1]
    non_conflict = [r for r in all_records if r["conflict"] == 0]

    overall_acc = sum(r["correct"] for r in all_records) / total
    conflict_acc = sum(r["correct"] for r in conflict_records) / len(conflict_records) if conflict_records else 0
    non_conflict_acc = sum(r["correct"] for r in non_conflict) / len(non_conflict) if non_conflict else 0

    print(f"\n总样本数: {total}")
    print(f"冲突样本: {len(conflict_records)} ({len(conflict_records)/total:.1%})")
    print(f"非冲突样本: {len(non_conflict)} ({len(non_conflict)/total:.1%})")
    print(f"\n整体准确率: {overall_acc:.4f}")
    print(f"冲突样本准确率: {conflict_acc:.4f}")
    print(f"非冲突样本准确率: {non_conflict_acc:.4f}")

    # 可靠性分布
    print("\n--- 各模态可靠性分布 ---")
    for m in ALL_MODALITIES:
        key = f"rel_{m}"
        vals = [r[key] for r in all_records if r[key] >= 0]
        if vals:
            name = {"text": "文本", "visual": "视觉", "audio": "音频"}[m]
            print(f"  {name}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"min={np.min(vals):.4f} max={np.max(vals):.4f}")

    # 冲突样本中MRE选择了哪个模态
    print("\n--- 冲突样本中MRE的模态选择 ---")
    pick_counts = {}
    pick_correct = {}
    for r in conflict_records:
        path = r["path"]
        if path.startswith("pick_"):
            m = path.split("pick_")[1]
            pick_counts[m] = pick_counts.get(m, 0) + 1
            if r["correct"]:
                pick_correct[m] = pick_correct.get(m, 0) + 1
    for m in ALL_MODALITIES:
        cnt = pick_counts.get(m, 0)
        corr = pick_correct.get(m, 0)
        acc = corr / cnt if cnt > 0 else 0
        name = {"text": "文本", "visual": "视觉", "audio": "音频"}[m]
        print(f"  选择{name}: {cnt}次 ({cnt/len(conflict_records):.1%}), 正确率: {acc:.4f}")

    # 按标签分析
    print("\n--- 各类别准确率 ---")
    for c in range(NUM_CLASSES):
        c_records = [r for r in all_records if r["label"] == c]
        if c_records:
            c_acc = sum(r["correct"] for r in c_records) / len(c_records)
            c_conflict = [r for r in c_records if r["conflict"] == 1]
            c_conf_acc = sum(r["correct"] for r in c_conflict) / len(c_conflict) if c_conflict else 0
            print(f"  {LABEL_NAMES[c]}: 总体={c_acc:.4f} 冲突={c_conf_acc:.4f} (n={len(c_records)}, 冲突n={len(c_conflict)})")

    # 保存详细记录
    if out_dir:
        csv_path = os.path.join(out_dir, "exp3_conflict_records.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "label", "pred", "correct", "conflict", "path",
                             "rel_text", "rel_visual", "rel_audio",
                             "pred_text", "pred_visual", "pred_audio", "n_available"])
            for r in all_records:
                writer.writerow([r["idx"], r["label"], r["pred"], r["correct"], r["conflict"],
                                 r["path"], f"{r['rel_text']:.4f}", f"{r['rel_visual']:.4f}",
                                 f"{r['rel_audio']:.4f}", r["pred_text"], r["pred_visual"],
                                 r["pred_audio"], r["n_available"]])

        # 汇总JSON
        summary = {
            "total_samples": total,
            "conflict_samples": len(conflict_records),
            "conflict_rate": len(conflict_records) / total,
            "overall_accuracy": overall_acc,
            "conflict_accuracy": conflict_acc,
            "non_conflict_accuracy": non_conflict_acc,
            "modality_reliability": {},
            "modality_selection_in_conflicts": {},
            "per_class_accuracy": {}
        }
        for m in ALL_MODALITIES:
            vals = [r[f"rel_{m}"] for r in all_records if r[f"rel_{m}"] >= 0]
            summary["modality_reliability"][m] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals))
            } if vals else {}
        for m in ALL_MODALITIES:
            cnt = pick_counts.get(m, 0)
            corr = pick_correct.get(m, 0)
            summary["modality_selection_in_conflicts"][m] = {
                "count": cnt, "accuracy": corr / cnt if cnt > 0 else 0
            }
        for c in range(NUM_CLASSES):
            c_records = [r for r in all_records if r["label"] == c]
            if c_records:
                summary["per_class_accuracy"][LABEL_NAMES[c]] = {
                    "accuracy": sum(r["correct"] for r in c_records) / len(c_records),
                    "n_samples": len(c_records)
                }
        with open(os.path.join(out_dir, "exp3_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return all_records


# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description="补充实验")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="13,42,2024")
    parser.add_argument("--out_dir", type=str, default="outputs/supplementary")
    parser.add_argument("--exp", type=str, default="all",
                        help="运行哪个实验: 1, 2, 3, 或 all")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"补充实验 | 样本数: {args.samples} | 种子: {seeds} | 设备: {device}")
    print(f"真实模型准确率: T={REAL_ACCURACIES['text']:.1%} V={REAL_ACCURACIES['visual']:.1%} A={REAL_ACCURACIES['audio']:.1%}")

    run_all = args.exp == "all"

    if run_all or args.exp == "1":
        exp1_missing_rate(args, seeds, device, out_dir)

    if run_all or args.exp == "2":
        exp2_hyperparams(args, seeds, device, out_dir)

    if run_all or args.exp == "3":
        exp3_conflict_analysis(args, seeds, device, out_dir)

    print("\n" + "=" * 70)
    print("所有补充实验完成！")
    if out_dir:
        print(f"结果保存在: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
