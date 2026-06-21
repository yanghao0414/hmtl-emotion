# -*- coding: utf-8 -*-
"""
综合场景实验：模拟真实部署环境

真实场景中，不是"所有模态同时加噪声"，而是"随机某个模态被干扰"。
本实验模拟：每个样本随机选一个模态加噪声（或不加），计算MRE vs 加权投票的期望收益。

场景设定：
  - 场景A：干净（0%噪声）
  - 场景B：随机单模态噪声（每样本随机选一个模态加20%噪声）
  - 场景C：随机单模态噪声（每样本随机选一个模态加40%噪声）
  - 场景D：混合（50%样本干净 + 50%样本随机单模态噪声30%）
  - 场景E：极端混合（30%干净 + 40%单模态噪声30% + 30%双模态噪声20%）

Run:
  python exp_mixed_scenario.py --samples 5000 --seeds 13,42,2024,99,7
"""
import argparse
import random
import os
import csv


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

NUM_CLASSES = 4
ALL_MODALITIES = ["text", "visual", "audio"]

CLASS_MODALITY_ACC = {
    0: {"text": 0.85, "visual": 0.70, "audio": 0.80},
    1: {"text": 0.75, "visual": 0.55, "audio": 0.82},
    2: {"text": 0.80, "visual": 0.50, "audio": 0.65},
    3: {"text": 0.70, "visual": 0.72, "audio": 0.70},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MixedScenarioDataset(Dataset):
    """每个样本独立决定噪声模式"""
    def __init__(self, n_samples=5000, missing_rate=0.1, seed=42,
                 scenario="mixed_D"):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        self.missing_rate = missing_rate
        self.scenario = scenario

        self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)
        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            # 决定本样本的噪声配置
            noise_cfg = self._sample_noise_config()
            modalities = {}
            for m in ALL_MODALITIES:
                if self.missing_rate > 0 and np.random.rand() < self.missing_rate:
                    modalities[m] = {"available": False}
                    continue
                acc = CLASS_MODALITY_ACC[y][m]
                nr = noise_cfg.get(m, 0.0)
                if nr > 0:
                    acc = acc * (1 - nr) + (1.0 / NUM_CLASSES) * nr
                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong))
                logits = self._gen_logits(NUM_CLASSES, pred, peak=3.0)
                if nr > 0:
                    logits += np.random.normal(0, nr * 2, size=logits.shape)
                probs = self._softmax(logits)
                conf = float(np.max(probs))
                modalities[m] = {
                    "available": True, "pred_idx": int(pred),
                    "probs": probs.astype(np.float32),
                    "logits": logits.astype(np.float32),
                    "confidence": conf,
                }
            self.samples.append({"label": y, "modalities": modalities})

    def _sample_noise_config(self):
        s = self.scenario
        if s == "clean":
            return {}
        elif s == "random_single_20":
            m = random.choice(ALL_MODALITIES)
            return {m: 0.20}
        elif s == "random_single_40":
            m = random.choice(ALL_MODALITIES)
            return {m: 0.40}
        elif s == "mixed_D":
            # 50%干净 + 50%随机单模态噪声30%
            if random.random() < 0.5:
                return {}
            else:
                m = random.choice(ALL_MODALITIES)
                return {m: 0.30}
        elif s == "mixed_E":
            # 30%干净 + 40%单模态噪声30% + 30%双模态噪声20%
            r = random.random()
            if r < 0.3:
                return {}
            elif r < 0.7:
                m = random.choice(ALL_MODALITIES)
                return {m: 0.30}
            else:
                ms = random.sample(ALL_MODALITIES, 2)
                return {ms[0]: 0.20, ms[1]: 0.20}
        elif s == "mixed_F":
            # 20%干净 + 30%单模态噪声40% + 30%双模态噪声30% + 20%全模态噪声20%
            r = random.random()
            if r < 0.2:
                return {}
            elif r < 0.5:
                m = random.choice(ALL_MODALITIES)
                return {m: 0.40}
            elif r < 0.8:
                ms = random.sample(ALL_MODALITIES, 2)
                return {ms[0]: 0.30, ms[1]: 0.30}
            else:
                return {"text": 0.20, "visual": 0.20, "audio": 0.20}
        return {}

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


# ============ MRE (same as exp2_lambda_v2) ============
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
    def __init__(self, in_dim=6, h_dim=32, out_dim=1, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim // 2), nn.ReLU(), nn.Dropout(dropout),
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
    ent_losses = []
    for m in ALL_MODALITIES:
        if m in reliabilities:
            r = reliabilities[m].view(-1)
            r = torch.clamp(r, 1e-6, 1 - 1e-6)
            ent = -(r * torch.log(r) + (1 - r) * torch.log(1 - r))
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
        return 0
    if len(set(preds.values())) == 1:
        return list(preds.values())[0]
    best_m, best_r = None, -1.0
    for m, p in preds.items():
        r = float(reliabilities.get(m, 0.0))
        if r > best_r:
            best_r = r
            best_m = m
    return preds[best_m]

def weighted_vote(sample):
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

def majority_vote(sample):
    from collections import Counter
    preds = []
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds.append(int(info["pred_idx"]))
    if not preds:
        return 0
    return Counter(preds).most_common(1)[0][0]

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

def train_mre(train_set, val_set, epochs=15, lr=3e-3, lambda_ent=0.2,
              batch_size=64, device=None):
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
            loss = mre_bce_balanced(rel, corr_b) + lambda_ent * entropy_regularization(rel)
            loss.backward()
            optimizer.step()
        val_m = eval_method(val_set, mre, device, "mre")
        if val_m > best_acc:
            best_acc = val_m
            best_state = {k: v.clone() for k, v in mre.state_dict().items()}
    if best_state:
        mre.load_state_dict(best_state)
    return mre

def eval_method(dataset, mre_model, device, method="mre"):
    if method == "mre":
        mre_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if method == "mre":
                feats_t = {}
                for m in ALL_MODALITIES:
                    f, _, _, avail = features_for_modality(sample, m)
                    if avail:
                        feats_t[m] = torch.from_numpy(f).float().unsqueeze(0).to(device)
                rel = {}
                if feats_t:
                    r = mre_model(feats_t)
                    rel = {k: float(v.item()) for k, v in r.items()}
                pred = fuse_with_reliability(sample, rel)
            elif method == "weighted":
                pred = weighted_vote(sample)
            elif method == "majority":
                pred = majority_vote(sample)
            y_true.append(int(sample["label"]))
            y_pred.append(int(pred))
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    return acc

def eval_all(dataset, mre_model, device):
    mre_acc = eval_method(dataset, mre_model, device, "mre")
    wt_acc = eval_method(dataset, None, device, "weighted")
    mj_acc = eval_method(dataset, None, device, "majority")
    return mre_acc, wt_acc, mj_acc


# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_ent", type=float, default=0.2)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="13,42,2024,99,7")
    parser.add_argument("--out_dir", type=str, default="outputs/mixed_scenario")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    scenarios = [
        ("clean",           "A: 干净环境"),
        ("random_single_20","B: 随机单模态噪声20%"),
        ("random_single_40","C: 随机单模态噪声40%"),
        ("mixed_D",         "D: 50%干净+50%单模态噪声30%"),
        ("mixed_E",         "E: 30%干净+40%单噪声30%+30%双噪声20%"),
        ("mixed_F",         "F: 20%干净+30%单噪声40%+30%双噪声30%+20%全噪声20%"),
    ]

    print("=" * 70)
    print("综合场景实验：模拟真实部署环境下MRE vs 加权投票 vs 多数投票")
    print(f"样本数: {args.samples} | 种子: {seeds} | λ_ent: {args.lambda_ent}")
    print("=" * 70)

    all_results = {}
    print(f"\n{'场景':<45} {'MRE':>12} {'加权投票':>12} {'多数投票':>12} {'MRE-加权':>10}")
    print("-" * 95)

    for scenario_key, scenario_name in scenarios:
        mre_accs, wt_accs, mj_accs = [], [], []
        for seed in seeds:
            ds = MixedScenarioDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                       seed=seed, scenario=scenario_key)
            train_set, val_set, test_set = split_dataset(ds)
            mre = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                           lambda_ent=args.lambda_ent, device=device)
            m_acc, w_acc, j_acc = eval_all(test_set, mre, device)
            mre_accs.append(m_acc)
            wt_accs.append(w_acc)
            mj_accs.append(j_acc)

        r = {
            "mre_mean": statistics.mean(mre_accs),
            "mre_std": statistics.stdev(mre_accs) if len(mre_accs) >= 2 else 0,
            "wt_mean": statistics.mean(wt_accs),
            "wt_std": statistics.stdev(wt_accs) if len(wt_accs) >= 2 else 0,
            "mj_mean": statistics.mean(mj_accs),
            "mj_std": statistics.stdev(mj_accs) if len(mj_accs) >= 2 else 0,
        }
        all_results[scenario_key] = r
        delta = r["mre_mean"] - r["wt_mean"]
        print(f"  {scenario_name:<43} {r['mre_mean']:.4f}±{r['mre_std']:.4f}"
              f"  {r['wt_mean']:.4f}±{r['wt_std']:.4f}"
              f"  {r['mj_mean']:.4f}±{r['mj_std']:.4f}"
              f"  {delta:+.4f}")

    # 计算期望收益
    print("\n" + "=" * 70)
    print("期望收益分析")
    print("=" * 70)
    deltas = []
    for key, name in scenarios:
        r = all_results[key]
        d = r["mre_mean"] - r["wt_mean"]
        deltas.append(d)
        print(f"  {name:<43} Δ(MRE-加权) = {d:+.4f} ({d*100:+.2f}pp)")
    avg_delta = statistics.mean(deltas)
    print(f"\n  平均期望收益: {avg_delta:+.4f} ({avg_delta*100:+.2f}pp)")
    # 只看有噪声的场景
    noisy_deltas = deltas[1:]  # 排除clean
    avg_noisy = statistics.mean(noisy_deltas)
    print(f"  噪声场景平均收益: {avg_noisy:+.4f} ({avg_noisy*100:+.2f}pp)")

    # 保存
    if args.out_dir:
        csv_path = os.path.join(args.out_dir, "mixed_scenario_results.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "scenario_name", "mre_mean", "mre_std",
                             "weighted_mean", "weighted_std", "majority_mean", "majority_std",
                             "delta_mre_weighted"])
            for key, name in scenarios:
                r = all_results[key]
                writer.writerow([key, name,
                                 f"{r['mre_mean']:.4f}", f"{r['mre_std']:.4f}",
                                 f"{r['wt_mean']:.4f}", f"{r['wt_std']:.4f}",
                                 f"{r['mj_mean']:.4f}", f"{r['mj_std']:.4f}",
                                 f"{r['mre_mean']-r['wt_mean']:.4f}"])
        print(f"\n已保存: {csv_path}")


if __name__ == "__main__":
    main()
