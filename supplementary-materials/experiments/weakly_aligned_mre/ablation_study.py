# -*- coding: utf-8 -*-
"""
消融实验：单模态 vs 双模态 vs 三模态 + MRE融合对比
使用真实模型准确率模拟各模态预测，评估不同模态组合的性能

实验设计：
1. 单模态: T / V / A
2. 双模态: T+V / T+A / V+A
3. 三模态: T+V+A (baseline: confidence) / T+V+A (MRE)

Run:
  python ablation_study.py --samples 5000 --epochs 10 --seeds 13,42,2024,100,777
"""
import argparse
import random
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import statistics

NUM_CLASSES = 4
ALL_MODALITIES = ["text", "visual", "audio"]

# ============ 真实模型准确率 ============
REAL_ACCURACIES = {
    "text":   0.789,   # SMP2020 4类 78.7% (用公开数据集的，更可信)
    "visual": 0.629,   # V4 EfficientNet-B2 4类 62.9%
    "audio":  0.754,   # V2 Wav2Vec2 4类 75.4%
}

# 4类标签名
LABEL_NAMES = ['积极', '激活消极', '非激活消极', '平静']


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AblationDataset(Dataset):
    """
    模拟数据集：基于真实模型准确率生成各模态预测
    active_modalities: 指定哪些模态可用
    """
    def __init__(self, n_samples: int = 5000,
                 modality_acc: Dict[str, float] = None,
                 active_modalities: List[str] = None,
                 missing_rate: float = 0.0,
                 seed: int = 42):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        if modality_acc is None:
            modality_acc = REAL_ACCURACIES.copy()
        self.modality_acc = modality_acc
        self.active = active_modalities or ALL_MODALITIES
        self.missing_rate = missing_rate

        # 生成均匀分布的标签
        self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)
        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            modalities = {}
            for m in ALL_MODALITIES:
                if m not in self.active:
                    modalities[m] = {"available": False}
                    continue
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
                    "available": True,
                    "pred_idx": int(pred),
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


def majority_vote(sample):
    """多数投票融合"""
    preds = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds[m] = int(info["pred_idx"])
    if not preds:
        return 0
    # 统计投票
    votes = {}
    for p in preds.values():
        votes[p] = votes.get(p, 0) + 1
    max_votes = max(votes.values())
    candidates = [p for p, v in votes.items() if v == max_votes]
    if len(candidates) == 1:
        return candidates[0]
    # 平票时选置信度最高的模态
    best_conf, best_pred = -1, candidates[0]
    for m, p in preds.items():
        if p in candidates:
            conf = sample["modalities"][m].get("confidence", 0)
            if conf > best_conf:
                best_conf = conf
                best_pred = p
    return best_pred


def weighted_vote(sample, weights=None):
    """加权投票融合"""
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


def evaluate_single_modality(dataset, modality):
    """评估单模态性能"""
    y_true, y_pred = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        info = sample["modalities"].get(modality, {"available": False})
        if info.get("available", False):
            y_true.append(int(sample["label"]))
            y_pred.append(int(info["pred_idx"]))
    if not y_true:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n_samples": 0}
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1, "n_samples": len(y_true)}


def evaluate_fusion(dataset, method="majority"):
    """评估融合方法"""
    y_true, y_pred = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        if method == "majority":
            pred = majority_vote(sample)
        elif method == "weighted":
            pred = weighted_vote(sample)
        elif method == "confidence":
            rel = {}
            for m in ALL_MODALITIES:
                info = sample["modalities"].get(m, {"available": False})
                if info.get("available", False):
                    rel[m] = float(info["confidence"])
            pred, _ = fuse_with_reliability(sample, rel)
        else:
            pred = majority_vote(sample)
        y_true.append(int(sample["label"]))
        y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": mf1}


def evaluate_mre(dataset, mre_model, device):
    """评估MRE融合"""
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


def train_mre(train_set, val_set, args, device):
    """训练MRE模型"""
    mre = MRE(in_dim=6, h_dim=8).to(device)
    optimizer = optim.AdamW(mre.parameters(), lr=args.lr)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)

    best_acc = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        mre.train()
        running = []
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
            loss = args.lambda_mre * mre_bce_balanced(rel, corr_b)
            loss.backward()
            optimizer.step()
            running.append(float(loss.item()))

        # 验证
        val_metrics = evaluate_mre(val_set, mre, device)
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.clone() for k, v in mre.state_dict().items()}

    if best_state:
        mre.load_state_dict(best_state)
    return mre


def run_ablation(args, seed):
    """运行一次完整的消融实验"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    results = {}

    # ============ 1. 单模态实验 ============
    for m in ALL_MODALITIES:
        ds = AblationDataset(n_samples=args.samples, active_modalities=[m], seed=seed)
        n = len(ds)
        test_idx = list(range(int(0.85 * n), n))
        test_set = torch.utils.data.Subset(ds, test_idx)
        metrics = evaluate_single_modality(test_set, m)
        key = f"single_{m}"
        results[key] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    # ============ 2. 双模态实验 ============
    for combo in combinations(ALL_MODALITIES, 2):
        combo_list = list(combo)
        ds = AblationDataset(n_samples=args.samples, active_modalities=combo_list, seed=seed)
        n = len(ds)
        train_idx = list(range(int(0.7 * n)))
        val_idx = list(range(int(0.7 * n), int(0.85 * n)))
        test_idx = list(range(int(0.85 * n), n))
        train_set = torch.utils.data.Subset(ds, train_idx)
        val_set = torch.utils.data.Subset(ds, val_idx)
        test_set = torch.utils.data.Subset(ds, test_idx)

        key = f"dual_{'_'.join(combo_list)}"
        # 多数投票
        results[f"{key}_majority"] = evaluate_fusion(test_set, "majority")
        # 加权投票
        results[f"{key}_weighted"] = evaluate_fusion(test_set, "weighted")
        # 置信度融合
        results[f"{key}_confidence"] = evaluate_fusion(test_set, "confidence")
        # MRE融合
        mre = train_mre(train_set, val_set, args, device)
        results[f"{key}_mre"] = evaluate_mre(test_set, mre, device)

    # ============ 3. 三模态实验 ============
    ds = AblationDataset(n_samples=args.samples, active_modalities=ALL_MODALITIES,
                         missing_rate=args.missing_rate, seed=seed)
    n = len(ds)
    train_idx = list(range(int(0.7 * n)))
    val_idx = list(range(int(0.7 * n), int(0.85 * n)))
    test_idx = list(range(int(0.85 * n), n))
    train_set = torch.utils.data.Subset(ds, train_idx)
    val_set = torch.utils.data.Subset(ds, val_idx)
    test_set = torch.utils.data.Subset(ds, test_idx)

    results["triple_majority"] = evaluate_fusion(test_set, "majority")
    results["triple_weighted"] = evaluate_fusion(test_set, "weighted")
    results["triple_confidence"] = evaluate_fusion(test_set, "confidence")
    mre = train_mre(train_set, val_set, args, device)
    results["triple_mre"] = evaluate_mre(test_set, mre, device)

    return results


def print_ablation_table(agg_results):
    """打印消融实验结果表格"""
    print("\n" + "=" * 80)
    print("消融实验结果 (4类分类)")
    print("=" * 80)

    # 表头
    print(f"\n{'实验配置':<35} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 60)

    # 单模态
    print("\n--- 单模态 ---")
    for m in ALL_MODALITIES:
        key = f"single_{m}"
        r = agg_results[key]
        name_map = {"text": "文本(T)", "visual": "视觉(V)", "audio": "音频(A)"}
        print(f"  {name_map[m]:<33} {r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f} {r['macro_f1']['mean']:.4f}±{r['macro_f1']['std']:.4f}")

    # 双模态
    print("\n--- 双模态 ---")
    for combo in combinations(ALL_MODALITIES, 2):
        combo_list = list(combo)
        base_key = f"dual_{'_'.join(combo_list)}"
        name_map = {"text": "T", "visual": "V", "audio": "A"}
        combo_name = "+".join(name_map[m] for m in combo_list)
        for method in ["majority", "weighted", "confidence", "mre"]:
            key = f"{base_key}_{method}"
            r = agg_results[key]
            method_cn = {"majority": "多数投票", "weighted": "加权投票", "confidence": "置信度", "mre": "MRE"}[method]
            print(f"  {combo_name} ({method_cn}){'':<15} {r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f} {r['macro_f1']['mean']:.4f}±{r['macro_f1']['std']:.4f}")

    # 三模态
    print("\n--- 三模态 (T+V+A) ---")
    for method in ["majority", "weighted", "confidence", "mre"]:
        key = f"triple_{method}"
        r = agg_results[key]
        method_cn = {"majority": "多数投票", "weighted": "加权投票", "confidence": "置信度", "mre": "MRE(本文)"}[method]
        marker = " ***" if method == "mre" else ""
        print(f"  T+V+A ({method_cn}){'':<15} {r['accuracy']['mean']:.4f}±{r['accuracy']['std']:.4f} {r['macro_f1']['mean']:.4f}±{r['macro_f1']['std']:.4f}{marker}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="消融实验")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="13,42,2024",
                        help="逗号分隔的随机种子列表")
    parser.add_argument("--out_dir", type=str, default="outputs/ablation")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"消融实验 | 样本数: {args.samples} | 种子: {seeds}")
    print(f"真实模型准确率: T={REAL_ACCURACIES['text']:.1%} V={REAL_ACCURACIES['visual']:.1%} A={REAL_ACCURACIES['audio']:.1%}")
    print(f"缺失率: {args.missing_rate:.1%}")

    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_ablation(args, seed)
        all_results.append(results)
        # 简要打印
        print(f"  单模态: T={results['single_text']['accuracy']:.4f} V={results['single_visual']['accuracy']:.4f} A={results['single_audio']['accuracy']:.4f}")
        print(f"  三模态MRE: {results['triple_mre']['accuracy']:.4f}")

    # 聚合
    all_keys = all_results[0].keys()
    agg = {}
    for key in all_keys:
        accs = [r[key]["accuracy"] for r in all_results]
        f1s = [r[key]["macro_f1"] for r in all_results]
        agg[key] = {
            "accuracy": {
                "mean": float(statistics.mean(accs)),
                "std": float(statistics.stdev(accs)) if len(accs) >= 2 else 0.0
            },
            "macro_f1": {
                "mean": float(statistics.mean(f1s)),
                "std": float(statistics.stdev(f1s)) if len(f1s) >= 2 else 0.0
            }
        }

    print_ablation_table(agg)

    # 保存结果
    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # JSON
        save_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "args": vars(args),
            "real_accuracies": REAL_ACCURACIES,
            "seeds": seeds,
            "aggregate": agg,
            "per_seed": all_results
        }
        with open(os.path.join(out_dir, "ablation_results.json"), "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        # CSV for paper
        csv_path = os.path.join(out_dir, "ablation_table.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["配置", "融合方法", "Accuracy_mean", "Accuracy_std", "Macro-F1_mean", "Macro-F1_std"])
            # 单模态
            for m in ALL_MODALITIES:
                key = f"single_{m}"
                r = agg[key]
                name = {"text": "文本(T)", "visual": "视觉(V)", "audio": "音频(A)"}[m]
                writer.writerow([name, "-", f"{r['accuracy']['mean']:.4f}", f"{r['accuracy']['std']:.4f}",
                                 f"{r['macro_f1']['mean']:.4f}", f"{r['macro_f1']['std']:.4f}"])
            # 双模态
            for combo in combinations(ALL_MODALITIES, 2):
                combo_list = list(combo)
                base_key = f"dual_{'_'.join(combo_list)}"
                name_map = {"text": "T", "visual": "V", "audio": "A"}
                combo_name = "+".join(name_map[m] for m in combo_list)
                for method in ["majority", "weighted", "confidence", "mre"]:
                    key = f"{base_key}_{method}"
                    r = agg[key]
                    method_cn = {"majority": "多数投票", "weighted": "加权投票", "confidence": "置信度", "mre": "MRE"}[method]
                    writer.writerow([combo_name, method_cn, f"{r['accuracy']['mean']:.4f}", f"{r['accuracy']['std']:.4f}",
                                     f"{r['macro_f1']['mean']:.4f}", f"{r['macro_f1']['std']:.4f}"])
            # 三模态
            for method in ["majority", "weighted", "confidence", "mre"]:
                key = f"triple_{method}"
                r = agg[key]
                method_cn = {"majority": "多数投票", "weighted": "加权投票", "confidence": "置信度", "mre": "MRE(本文)"}[method]
                writer.writerow(["T+V+A", method_cn, f"{r['accuracy']['mean']:.4f}", f"{r['accuracy']['std']:.4f}",
                                 f"{r['macro_f1']['mean']:.4f}", f"{r['macro_f1']['std']:.4f}"])

        print(f"\n结果已保存到: {out_dir}")


if __name__ == "__main__":
    main()
