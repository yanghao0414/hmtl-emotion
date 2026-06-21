# -*- coding: utf-8 -*-
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 4
MODALITIES = ["text", "visual", "audio"]


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IemocapWeakDataset(Dataset):
    def __init__(self, data_csv: str = "", n_samples: int = 0, missing_rate: float = 0.1, seed: int = 42):
        super().__init__()
        set_seed(seed)
        self.samples = []
        if data_csv and os.path.isfile(data_csv):
            self._load_csv(data_csv)
        else:
            # demo fallback using synthetic generation
            n = n_samples if n_samples > 0 else 1000
            self._gen_synthetic(n, missing_rate)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def _parse_probs(self, s: str):
        try:
            parts = [float(x) for x in s.split(";")]
            arr = np.array(parts, dtype=np.float32)
            arr = np.clip(arr, 1e-8, 1.0)
            arr /= arr.sum()
            return arr
        except Exception:
            return None

    def _load_csv(self, path: str):
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = int(row.get("label4", -1))
                modalities = {}
                for m in MODALITIES:
                    avail = int(row.get(f"{m}_available", 1))
                    if not avail:
                        modalities[m] = {"available": False}
                        continue
                    pred = int(row.get(f"{m}_pred", -1))
                    conf = float(row.get(f"{m}_conf", 0.0))
                    probs_str = row.get(f"{m}_probs", "")
                    probs = self._parse_probs(probs_str) if probs_str else None
                    if probs is None:
                        probs = np.ones((NUM_CLASSES,), dtype=np.float32) * ((1.0 - conf) / (NUM_CLASSES - 1 + 1e-8))
                        max_idx = pred if pred >= 0 else int(np.argmax(probs))
                        probs[max_idx] = conf
                        probs = np.clip(probs, 1e-6, 1.0)
                        probs /= probs.sum()
                        pred = int(np.argmax(probs))
                    logits = np.log(probs + 1e-8)
                    modalities[m] = {
                        "available": True,
                        "pred_idx": int(pred),
                        "probs": probs.astype(np.float32),
                        "logits": logits.astype(np.float32),
                        "confidence": float(np.max(probs))
                    }
                self.samples.append({"label": label, "modalities": modalities})

    def _gen_logits(self, n_classes: int, pred_idx: int, peak: float = 3.0) -> np.ndarray:
        base = np.random.normal(0.0, 0.5, size=(n_classes,))
        base[pred_idx] += peak
        return base

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def _gen_synthetic(self, n_samples: int, missing_rate: float):
        self.samples = []
        modality_acc = {"text": 0.60, "visual": 0.58, "audio": 0.62}
        labels = np.random.randint(0, NUM_CLASSES, size=(n_samples,)).astype(np.int64)
        for i in range(n_samples):
            y = int(labels[i])
            mods = {}
            for m in MODALITIES:
                available = (np.random.rand() > missing_rate)
                if not available:
                    mods[m] = {"available": False}
                    continue
                acc = modality_acc[m]
                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong))
                logits = self._gen_logits(NUM_CLASSES, pred_idx=pred, peak=3.0)
                probs = self._softmax(logits)
                conf = float(np.max(probs))
                mods[m] = {
                    "available": True,
                    "pred_idx": int(pred),
                    "probs": probs.astype(np.float32),
                    "logits": logits.astype(np.float32),
                    "confidence": conf,
                }
            self.samples.append({"label": y, "modalities": mods})


def compute_entropy(probs: np.ndarray) -> float:
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


def features_for_modality(sample: dict, modality: str) -> Tuple[np.ndarray, int, float, bool]:
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
    for m2 in MODALITIES:
        if m2 == modality:
            continue
        o = mods.get(m2, {"available": False})
        if o.get("available", False):
            others.append(int(o["pred_idx"]))
    agreement = 0.5 if len(others) == 0 else float(sum(1 for p in others if p == pred) / len(others))
    feat = np.array([conf, ent, logit_mean, logit_var, 0.0, agreement], dtype=np.float32)
    return feat, pred, conf, True


class MLP(nn.Module):
    def __init__(self, in_dim=6, h_dim=8, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)


class MRE(nn.Module):
    def __init__(self, in_dim=6, h_dim=8):
        super().__init__()
        self.text_mlp = MLP(in_dim, h_dim, 1)
        self.visual_mlp = MLP(in_dim, h_dim, 1)
        self.audio_mlp = MLP(in_dim, h_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        if "text" in feats:
            out["text"] = self.sigmoid(self.text_mlp(feats["text"]))
        if "visual" in feats:
            out["visual"] = self.sigmoid(self.visual_mlp(feats["visual"]))
        if "audio" in feats:
            out["audio"] = self.sigmoid(self.audio_mlp(feats["audio"]))
        return out


def mre_bce_balanced(reliabilities: Dict[str, torch.Tensor], correctness: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for m in MODALITIES:
        if (m in reliabilities) and (m in correctness):
            r = reliabilities[m].view(-1)
            y = correctness[m].view(-1)
            pos_rate = torch.clamp(y.mean(), 1e-6, 1 - 1e-6)
            w_pos = 0.5 / pos_rate
            w_neg = 0.5 / (1 - pos_rate)
            r = torch.clamp(r, 1e-6, 1 - 1e-6)
            loss = - (w_pos * y * torch.log(r) + w_neg * (1 - y) * torch.log(1 - r))
            losses.append(loss.mean())
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def fuse_with_reliability(sample: dict, reliabilities: Dict[str, float]) -> Tuple[int, str]:
    preds = {}
    for m in MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds[m] = int(info["pred_idx"])
    if len(preds) == 0:
        return 0, "no_modality"
    if len(set(preds.values())) == 1:
        p = list(preds.values())[0]
        return p, "unanimous"
    best_m = None
    best_r = -1.0
    for m, p in preds.items():
        r = float(reliabilities.get(m, 0.0))
        if r > best_r:
            best_r = r
            best_m = m
    return preds[best_m], f"pick_{best_m}_by_r={best_r:.3f}"


def macro_f1(y_true: List[int], y_pred: List[int], n_classes: int = NUM_CLASSES) -> float:
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


def evaluate(dataset: Dataset, mre: MRE, device: torch.device) -> Dict[str, float]:
    mre.eval()
    y_true, y_pred = [], []
    conflict_correct = 0
    conflict_total = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            feats_t = {}
            rel = {}
            for m in MODALITIES:
                f, _, _, avail = features_for_modality(sample, m)
                if avail:
                    ft = torch.from_numpy(f).float().unsqueeze(0).to(device)
                    feats_t[m] = ft
            if feats_t:
                r = mre(feats_t)
                rel = {k: float(v.item()) for k, v in r.items()}
            final_pred, path = fuse_with_reliability(sample, rel)
            y_true.append(int(sample["label"]))
            y_pred.append(int(final_pred))
            preds = [int(info["pred_idx"]) for info in sample["modalities"].values() if info.get("available", False)]
            if len(preds) >= 2 and len(set(preds)) > 1:
                conflict_total += 1
                if final_pred == int(sample["label"]):
                    conflict_correct += 1
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred, NUM_CLASSES)
    conflict_acc = conflict_correct / conflict_total if conflict_total > 0 else 0.0
    return {"accuracy": acc, "macro_f1": mf1, "conflict_accuracy": conflict_acc}


def evaluate_confidence_baseline(dataset: Dataset) -> Dict[str, float]:
    y_true, y_pred = [], []
    conflict_correct = 0
    conflict_total = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        rel = {}
        for m in MODALITIES:
            info = sample["modalities"].get(m, {"available": False})
            if info.get("available", False):
                rel[m] = float(info.get("confidence", 0.0))
        final_pred, path = fuse_with_reliability(sample, rel)
        y_true.append(int(sample["label"]))
        y_pred.append(int(final_pred))
        preds = [int(info["pred_idx"]) for info in sample["modalities"].values() if info.get("available", False)]
        if len(preds) >= 2 and len(set(preds)) > 1:
            conflict_total += 1
            if final_pred == int(sample["label"]):
                conflict_correct += 1
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred, NUM_CLASSES)
    conflict_acc = conflict_correct / conflict_total if conflict_total > 0 else 0.0
    return {"accuracy": acc, "macro_f1": mf1, "conflict_accuracy": conflict_acc}


def _maybe_save_logs(out_dir: str, logs: dict):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    if "epochs" in logs and logs["epochs"]:
        csv_path = os.path.join(out_dir, "val_curve.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_mre_loss,val_acc,val_macro_f1,val_conflict_acc\n")
            for e in logs["epochs"]:
                f.write(f"{e['epoch']},{e['train_mre_loss']:.6f},{e['val']['accuracy']:.6f},{e['val']['macro_f1']:.6f},{e['val']['conflict_accuracy']:.6f}\n")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    full = IemocapWeakDataset(data_csv=args.data_csv, n_samples=args.samples, missing_rate=args.missing_rate, seed=args.seed)
    n = len(full)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    idxs = list(range(n))
    import random
    random.shuffle(idxs)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    subset = lambda idlist: torch.utils.data.Subset(full, idlist)
    train_set, val_set, test_set = subset(train_idx), subset(val_idx), subset(test_idx)

    mre = MRE(in_dim=6, h_dim=8).to(device)
    optim_mre = optim.AdamW(mre.parameters(), lr=args.lr)

    out_dir = args.out_dir
    logs = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "args": vars(args),
        "baseline": {},
        "epochs": []
    }

    val_conf_base = evaluate_confidence_baseline(val_set)
    test_conf_base = evaluate_confidence_baseline(test_set)
    print("Baseline (confidence-as-reliability):")
    print(f"- Val  | Acc: {val_conf_base['accuracy']:.4f} | MF1: {val_conf_base['macro_f1']:.4f} | ConflictAcc: {val_conf_base['conflict_accuracy']:.4f}")
    print(f"- Test | Acc: {test_conf_base['accuracy']:.4f} | MF1: {test_conf_base['macro_f1']:.4f} | ConflictAcc: {test_conf_base['conflict_accuracy']:.4f}")
    logs["baseline"] = {"val": val_conf_base, "test": test_conf_base}
    _maybe_save_logs(out_dir, logs)

    print("Stage A: no encoder training; proceed to Stage B (MRE)")

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    for epoch in range(1, args.epochs + 1):
        mre.train()
        running = []
        for batch in loader:
            optim_mre.zero_grad()
            feats_t = {"text": [], "visual": [], "audio": []}
            corr_t = {"text": [], "visual": [], "audio": []}
            any_available = {"text": False, "visual": False, "audio": False}
            for sample in batch:
                label = int(sample["label"])
                for m in MODALITIES:
                    f, pred, conf, avail = features_for_modality(sample, m)
                    if avail:
                        feats_t[m].append(torch.from_numpy(f).float())
                        corr_t[m].append(torch.tensor(1.0 if pred == label else 0.0, dtype=torch.float32))
                        any_available[m] = True
            feats_b = {}
            corr_b = {}
            for m in MODALITIES:
                if any_available[m]:
                    feats_b[m] = torch.stack(feats_t[m], dim=0).to(device)
                    corr_b[m] = torch.stack(corr_t[m], dim=0).to(device)
            if not feats_b:
                continue
            rel = mre(feats_b)
            loss_mre = mre_bce_balanced(rel, corr_b)
            total_loss = args.lambda_mre * loss_mre
            total_loss.backward()
            optim_mre.step()
            running.append(float(total_loss.item()))
        train_loss = float(np.mean(running)) if running else 0.0
        val_metrics = evaluate(val_set, mre, device)
        print(f"Epoch {epoch:02d} | L_mre: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val MF1: {val_metrics['macro_f1']:.4f} | Val ConflictAcc: {val_metrics['conflict_accuracy']:.4f}")
        logs["epochs"].append({
            "epoch": epoch,
            "train_mre_loss": train_loss,
            "val": val_metrics
        })
        _maybe_save_logs(out_dir, logs)

    test_mre = evaluate(test_set, mre, device)
    print("\nFinal Test Metrics:")
    print(f"- Confidence Baseline | Acc: {test_conf_base['accuracy']:.4f} | MF1: {test_conf_base['macro_f1']:.4f} | ConflictAcc: {test_conf_base['conflict_accuracy']:.4f}")
    print(f"- MRE (weakly-supervised) | Acc: {test_mre['accuracy']:.4f} | MF1: {test_mre['macro_f1']:.4f} | ConflictAcc: {test_mre['conflict_accuracy']:.4f}")
    logs["test_mre"] = test_mre
    _maybe_save_logs(out_dir, logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="", help="Path to IEMOCAP weak CSV (optional). If missing, use synthetic demo fallback.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=1000, help="Used only for demo fallback.")
    parser.add_argument("--missing_rate", type=float, default=0.1, help="Used only for demo fallback.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    set_seed(args.seed)
    train(args)
