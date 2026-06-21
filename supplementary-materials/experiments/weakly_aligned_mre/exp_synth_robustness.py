# -*- coding: utf-8 -*-
"""
合成数据鲁棒性验证：5次独立生成弱对齐数据集，验证结果波动范围

目的：证明合成机制不会引入系统性偏差，多次随机生成的结果一致。

Run:
  python exp_synth_robustness.py --samples 5000 --epochs 10
"""
import argparse, random, os, csv, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import stats as sp_stats

NUM_CLASSES = 4
ALL_MODALITIES = ["text", "visual", "audio"]
REAL_ACCURACIES = {"text": 0.789, "visual": 0.629, "audio": 0.754}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class SyntheticDataset(Dataset):
    def __init__(self, n_samples=5000, modality_acc=None, missing_rate=0.1,
                 seed=42, noise_config=None):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        if modality_acc is None:
            modality_acc = REAL_ACCURACIES.copy()
        self.modality_acc = modality_acc
        self.missing_rate = missing_rate
        self.noise_config = noise_config or {}
        self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)
        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            modalities = {}
            for m in ALL_MODALITIES:
                if self.missing_rate > 0 and np.random.rand() < self.missing_rate:
                    modalities[m] = {"available": False}; continue
                acc = self.modality_acc[m]
                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong))
                noise_rate = self.noise_config.get(m, 0.0)
                if noise_rate > 0 and np.random.rand() < noise_rate:
                    wrong = [c for c in range(NUM_CLASSES) if c != pred]
                    pred = int(np.random.choice(wrong))
                logits = self._gen_logits(NUM_CLASSES, pred, peak=3.0)
                if noise_rate > 0:
                    logits += np.random.normal(0, noise_rate * 2, size=logits.shape)
                probs = self._softmax(logits)
                conf = float(np.max(probs))
                modalities[m] = {"available": True, "pred_idx": int(pred),
                    "probs": probs.astype(np.float32), "logits": logits.astype(np.float32),
                    "confidence": conf}
            self.samples.append({"label": y, "modalities": modalities})

    def _gen_logits(self, n_classes, pred_idx, peak=3.0):
        base = np.random.normal(0.0, 0.5, size=(n_classes,)); base[pred_idx] += peak; return base
    def _softmax(self, x):
        x = x - np.max(x); e = np.exp(x); return e / np.sum(e)
    def __len__(self): return self.n
    def __getitem__(self, idx): return self.samples[idx]

def compute_entropy(probs):
    p = probs + 1e-12; return float(-np.sum(p * np.log(p)))

def features_for_modality(sample, modality):
    mods = sample["modalities"]
    s = mods.get(modality, {"available": False})
    if not s.get("available", False):
        return np.array([0.0, 2.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32), -1, 0.0, False
    probs = s["probs"]; pred = int(s["pred_idx"]); conf = float(s["confidence"])
    ent = compute_entropy(probs)
    logit_mean = float(np.mean(s["logits"])); logit_var = float(np.var(s["logits"]))
    others = []
    for m2 in ALL_MODALITIES:
        if m2 == modality: continue
        o = mods.get(m2, {"available": False})
        if o.get("available", False): others.append(int(o["pred_idx"]))
    agreement = float(sum(1 for p in others if p == pred) / len(others)) if others else 0.5
    feat = np.array([conf, ent, logit_mean, logit_var, 0.0, agreement], dtype=np.float32)
    return feat, pred, conf, True

class MLP(nn.Module):
    def __init__(self, in_dim=6, h_dim=8, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))
    def forward(self, x): return self.net(x)

class MRE(nn.Module):
    def __init__(self, in_dim=6, h_dim=8):
        super().__init__()
        self.text_mlp = MLP(in_dim, h_dim, 1)
        self.visual_mlp = MLP(in_dim, h_dim, 1)
        self.audio_mlp = MLP(in_dim, h_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feats):
        out = {}
        if "text" in feats: out["text"] = self.sigmoid(self.text_mlp(feats["text"]))
        if "visual" in feats: out["visual"] = self.sigmoid(self.visual_mlp(feats["visual"]))
        if "audio" in feats: out["audio"] = self.sigmoid(self.audio_mlp(feats["audio"]))
        return out

def mre_bce_balanced(reliabilities, correctness):
    losses = []
    for m in ALL_MODALITIES:
        if m in reliabilities and m in correctness:
            r = reliabilities[m].view(-1); y = correctness[m].view(-1)
            pos_rate = torch.clamp(y.mean(), 1e-6, 1 - 1e-6)
            w_pos = 0.5 / pos_rate; w_neg = 0.5 / (1 - pos_rate)
            r = torch.clamp(r, 1e-6, 1 - 1e-6)
            loss = -(w_pos * y * torch.log(r) + w_neg * (1 - y) * torch.log(1 - r))
            losses.append(loss.mean())
    if not losses: return torch.tensor(0.0)
    return torch.stack(losses).mean()

def fuse_with_reliability(sample, reliabilities):
    preds = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False): preds[m] = int(info["pred_idx"])
    if not preds: return 0
    if len(set(preds.values())) == 1: return list(preds.values())[0]
    best_m, best_r = None, -1.0
    for m, p in preds.items():
        r = float(reliabilities.get(m, 0.0))
        if r > best_r: best_r = r; best_m = m
    return preds[best_m]

def weighted_vote(sample):
    vote_scores = {}
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            pred = int(info["pred_idx"]); w = REAL_ACCURACIES.get(m, 0.5)
            vote_scores[pred] = vote_scores.get(pred, 0) + w
    if not vote_scores: return 0
    return max(vote_scores, key=vote_scores.get)

def majority_vote(sample):
    from collections import Counter
    preds = []
    for m in ALL_MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False): preds.append(int(info["pred_idx"]))
    if not preds: return 0
    return Counter(preds).most_common(1)[0][0]

def split_dataset(ds, train_r=0.7, val_r=0.15):
    n = len(ds); n_train = int(train_r * n); n_val = int(val_r * n)
    return (Subset(ds, list(range(n_train))),
            Subset(ds, list(range(n_train, n_train + n_val))),
            Subset(ds, list(range(n_train + n_val, n))))

def train_mre(train_set, val_set, epochs=10, lr=3e-3, lambda_mre=0.3,
              batch_size=64, device=None):
    if device is None: device = torch.device("cpu")
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
            if not feats_b: continue
            rel = mre(feats_b)
            loss = lambda_mre * mre_bce_balanced(rel, corr_b)
            loss.backward(); optimizer.step()
        mre.eval(); y_true, y_pred = [], []
        with torch.no_grad():
            for i in range(len(val_set)):
                sample = val_set[i]; feats_v = {}
                for m in ALL_MODALITIES:
                    f, _, _, avail = features_for_modality(sample, m)
                    if avail: feats_v[m] = torch.from_numpy(f).float().unsqueeze(0).to(device)
                rel_v = {}
                if feats_v:
                    r = mre(feats_v); rel_v = {k: float(v.item()) for k, v in r.items()}
                pred = fuse_with_reliability(sample, rel_v)
                y_true.append(int(sample["label"])); y_pred.append(int(pred))
        val_acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in mre.state_dict().items()}
    if best_state: mre.load_state_dict(best_state)
    return mre

def eval_acc(dataset, mre_model, device, method="mre"):
    if method == "mre": mre_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            if method == "mre":
                feats_t = {}
                for m in ALL_MODALITIES:
                    f, _, _, avail = features_for_modality(sample, m)
                    if avail: feats_t[m] = torch.from_numpy(f).float().unsqueeze(0).to(device)
                rel = {}
                if feats_t:
                    r = mre_model(feats_t); rel = {k: float(v.item()) for k, v in r.items()}
                pred = fuse_with_reliability(sample, rel)
            elif method == "weighted":
                pred = weighted_vote(sample)
            elif method == "majority":
                pred = majority_vote(sample)
            y_true.append(int(sample["label"])); y_pred.append(int(pred))
    return float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="outputs/synth_robustness")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 5次独立生成，每次用不同的数据生成种子
    gen_seeds = [1001, 2002, 3003, 4004, 5005]
    # 每次生成后，用固定的模型种子训练（隔离数据随机性 vs 模型随机性）
    model_seed = 42

    print("=" * 80)
    print("合成数据鲁棒性验证：5次独立生成弱对齐数据集")
    print("=" * 80)

    # 场景1: 干净数据（无噪声）
    # 场景2: 文本噪声30%
    # 场景3: 文本噪声50%
    scenarios = [
        ("干净数据", {}),
        ("文本噪声30%", {"text": 0.30}),
        ("文本噪声50%", {"text": 0.50}),
    ]

    all_rows = []

    for scenario_name, noise_cfg in scenarios:
        print(f"\n{'='*60}")
        print(f"场景: {scenario_name}")
        print(f"{'='*60}")

        mre_accs, wt_accs, mj_accs = [], [], []

        for gen_seed in gen_seeds:
            # 用gen_seed生成数据
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                   seed=gen_seed, noise_config=noise_cfg)
            # 用固定model_seed设置模型初始化
            set_seed(model_seed)
            train_set, val_set, test_set = split_dataset(ds)
            mre_model = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                                   lambda_mre=args.lambda_mre, device=device)
            m_acc = eval_acc(test_set, mre_model, device, "mre")
            w_acc = eval_acc(test_set, None, device, "weighted")
            j_acc = eval_acc(test_set, None, device, "majority")
            mre_accs.append(m_acc)
            wt_accs.append(w_acc)
            mj_accs.append(j_acc)
            print(f"  gen_seed={gen_seed}: MRE={m_acc:.4f}  加权={w_acc:.4f}  多数={j_acc:.4f}")

        mre_mean = statistics.mean(mre_accs)
        mre_std = statistics.stdev(mre_accs)
        wt_mean = statistics.mean(wt_accs)
        wt_std = statistics.stdev(wt_accs)
        mj_mean = statistics.mean(mj_accs)
        mj_std = statistics.stdev(mj_accs)
        mre_range = max(mre_accs) - min(mre_accs)
        wt_range = max(wt_accs) - min(wt_accs)

        print(f"\n  MRE:  {mre_mean:.4f} ± {mre_std:.4f}  (range: {mre_range:.4f} = {mre_range*100:.2f}pp)")
        print(f"  加权: {wt_mean:.4f} ± {wt_std:.4f}  (range: {wt_range:.4f} = {wt_range*100:.2f}pp)")
        print(f"  多数: {mj_mean:.4f} ± {mj_std:.4f}")

        all_rows.append({
            "scenario": scenario_name,
            "noise_config": str(noise_cfg),
            "mre_mean": mre_mean, "mre_std": mre_std, "mre_range_pp": mre_range * 100,
            "wt_mean": wt_mean, "wt_std": wt_std, "wt_range_pp": wt_range * 100,
            "mj_mean": mj_mean, "mj_std": mj_std,
            "n_generations": len(gen_seeds),
            "per_gen_mre": ";".join(f"{a:.4f}" for a in mre_accs),
            "per_gen_wt": ";".join(f"{a:.4f}" for a in wt_accs),
            "per_gen_mj": ";".join(f"{a:.4f}" for a in mj_accs),
        })

    # Save CSV
    csv_path = os.path.join(args.out_dir, "synth_robustness_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n已保存: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("汇总（可直接用于论文）")
    print("=" * 80)
    print(f"{'场景':<16} {'MRE均值±SD':>16} {'波动范围':>10} {'加权均值±SD':>16} {'波动范围':>10}")
    print("-" * 72)
    for r in all_rows:
        print(f"  {r['scenario']:<14} {r['mre_mean']:.4f}±{r['mre_std']:.4f}"
              f"  ±{r['mre_range_pp']:.2f}pp"
              f"  {r['wt_mean']:.4f}±{r['wt_std']:.4f}"
              f"  ±{r['wt_range_pp']:.2f}pp")


if __name__ == "__main__":
    main()
