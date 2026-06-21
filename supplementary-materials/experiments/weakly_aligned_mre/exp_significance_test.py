# -*- coding: utf-8 -*-
"""
统计显著性检验：对表3关键噪声对比点运行10种子paired t-test

重点检验：
  - 文本噪声 20%/30%/40%/50% 处 MRE vs 加权投票
  - 音频噪声 50% 处 MRE vs 加权投票

输出：per-seed accuracy + paired t-test p值 + 95% CI

Run:
  python exp_significance_test.py --samples 5000 --epochs 10
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

# ---- Dataset (same as adaptability_stability_experiments.py) ----
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

# ---- MRE Model ----
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

# ---- Training ----
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
        # val
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

# ============ Main ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="outputs/significance_test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    seeds_10 = [7, 13, 42, 99, 123, 456, 789, 2024, 3141, 9999]

    # Key comparison points
    test_cases = [
        ("text", 0.20, "文本噪声20%"),
        ("text", 0.30, "文本噪声30%"),
        ("text", 0.40, "文本噪声40%"),
        ("text", 0.50, "文本噪声50%"),
        ("audio", 0.50, "音频噪声50%"),
    ]

    print("=" * 80)
    print("统计显著性检验：MRE vs 加权投票 (10 seeds, paired t-test)")
    print("=" * 80)

    all_rows = []

    for target_m, noise_level, desc in test_cases:
        print(f"\n--- {desc} ---")
        noise_cfg = {target_m: noise_level}
        mre_accs, wt_accs = [], []

        for seed in seeds_10:
            ds = SyntheticDataset(n_samples=args.samples, missing_rate=args.missing_rate,
                                   seed=seed, noise_config=noise_cfg)
            train_set, val_set, test_set = split_dataset(ds)
            mre_model = train_mre(train_set, val_set, epochs=args.epochs, lr=args.lr,
                                   lambda_mre=args.lambda_mre, device=device)
            m_acc = eval_acc(test_set, mre_model, device, "mre")
            w_acc = eval_acc(test_set, None, device, "weighted")
            mre_accs.append(m_acc)
            wt_accs.append(w_acc)
            print(f"  seed={seed:>5d}: MRE={m_acc:.4f}  加权={w_acc:.4f}  Δ={m_acc-w_acc:+.4f}")

        # Paired t-test
        t_stat, p_value = sp_stats.ttest_rel(mre_accs, wt_accs)
        # Wilcoxon signed-rank (non-parametric alternative)
        try:
            w_stat, w_pvalue = sp_stats.wilcoxon(mre_accs, wt_accs)
        except ValueError:
            w_stat, w_pvalue = float('nan'), float('nan')

        # 95% CI for mean difference
        diffs = [m - w for m, w in zip(mre_accs, wt_accs)]
        mean_diff = statistics.mean(diffs)
        se_diff = statistics.stdev(diffs) / (len(diffs) ** 0.5)
        ci_low = mean_diff - 2.262 * se_diff  # t_{0.025, df=9}
        ci_high = mean_diff + 2.262 * se_diff

        # Cohen's d
        sd_diff = statistics.stdev(diffs)
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else float('inf')

        print(f"\n  MRE均值:  {statistics.mean(mre_accs):.4f} ± {statistics.stdev(mre_accs):.4f}")
        print(f"  加权均值: {statistics.mean(wt_accs):.4f} ± {statistics.stdev(wt_accs):.4f}")
        print(f"  均值差:   {mean_diff:+.4f} ({mean_diff*100:+.2f}pp)")
        print(f"  95% CI:   [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"  Paired t: t={t_stat:.4f}, p={p_value:.6f} {'***' if p_value<0.001 else '**' if p_value<0.01 else '*' if p_value<0.05 else 'n.s.'}")
        print(f"  Wilcoxon: W={w_stat:.1f}, p={w_pvalue:.6f}" if not np.isnan(w_pvalue) else "  Wilcoxon: N/A")
        print(f"  Cohen's d: {cohens_d:.3f}")

        all_rows.append({
            "condition": desc,
            "target_modality": target_m,
            "noise_level": noise_level,
            "mre_mean": statistics.mean(mre_accs),
            "mre_std": statistics.stdev(mre_accs),
            "wt_mean": statistics.mean(wt_accs),
            "wt_std": statistics.stdev(wt_accs),
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t_stat": t_stat,
            "p_value_ttest": p_value,
            "w_stat": w_stat if not np.isnan(w_stat) else "",
            "p_value_wilcoxon": w_pvalue if not np.isnan(w_pvalue) else "",
            "cohens_d": cohens_d,
            "n_seeds": len(seeds_10),
            "per_seed_mre": ";".join(f"{a:.4f}" for a in mre_accs),
            "per_seed_wt": ";".join(f"{a:.4f}" for a in wt_accs),
        })

    # Save CSV
    csv_path = os.path.join(args.out_dir, "significance_test_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n已保存: {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("汇总表（可直接用于论文）")
    print("=" * 80)
    print(f"{'条件':<16} {'MRE':>14} {'加权投票':>14} {'Δ':>10} {'95% CI':>22} {'p(t-test)':>12} {'显著性':>6}")
    print("-" * 100)
    for r in all_rows:
        sig = "***" if r["p_value_ttest"] < 0.001 else "**" if r["p_value_ttest"] < 0.01 else "*" if r["p_value_ttest"] < 0.05 else "n.s."
        print(f"  {r['condition']:<14} {r['mre_mean']:.4f}±{r['mre_std']:.4f}"
              f"  {r['wt_mean']:.4f}±{r['wt_std']:.4f}"
              f"  {r['mean_diff']:+.4f}"
              f"  [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]"
              f"  {r['p_value_ttest']:.6f}  {sig}")


if __name__ == "__main__":
    main()
