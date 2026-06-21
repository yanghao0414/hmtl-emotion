# -*- coding: utf-8 -*-
"""
Minimal reproducible training script for weakly-aligned multimodal fusion with MRE.
- Synthetic dataset (no alignment required)
- Decision-level fusion via reliability-driven learnable decision tree (simplified)
- Weak supervision for MRE: L_mre = BCE(reliability_m, correctness_m)

Run:
  python train_mre_weakly_aligned.py --epochs 5 --samples 2000
"""
import argparse
import math
import random
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import statistics

NUM_CLASSES = 4
MODALITIES = ["text", "visual", "audio"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticMultimodalDataset(Dataset):
    """
    Synthetic dataset that simulates per-modality predictions with controllable accuracies
    under a weakly-aligned/asynchronous setting (no cross-modal alignment).
    """
    def __init__(self, n_samples: int = 2000, 
                 modality_acc: Dict[str, float] = None,
                 missing_rate: float = 0.1,
                 seed: int = 42):
        super().__init__()
        set_seed(seed)
        self.n = n_samples
        self.missing_rate = missing_rate
        if modality_acc is None:
            modality_acc = {"text": 0.55, "visual": 0.60, "audio": 0.65}
        self.modality_acc = modality_acc

        self.labels = np.random.randint(0, NUM_CLASSES, size=(self.n,)).astype(np.int64)
        # Pre-generate per-modality predicted labels and logits distributions
        self.samples = []
        for i in range(self.n):
            y = int(self.labels[i])
            modalities = {}
            for m in MODALITIES:
                available = (np.random.rand() > self.missing_rate)
                if not available:
                    modalities[m] = {"available": False}
                    continue
                acc = self.modality_acc[m]
                # with prob acc, predict correct, else draw an incorrect class uniformly
                if np.random.rand() < acc:
                    pred = y
                else:
                    wrong_choices = [c for c in range(NUM_CLASSES) if c != y]
                    pred = int(np.random.choice(wrong_choices))
                # create a peaked logit distribution around pred
                logits = self._gen_logits(NUM_CLASSES, pred_idx=pred, peak=3.0)
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

    def _gen_logits(self, n_classes: int, pred_idx: int, peak: float = 3.0) -> np.ndarray:
        base = np.random.normal(0.0, 0.5, size=(n_classes,))
        base[pred_idx] += peak
        return base

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        return self.samples[idx]


def compute_entropy(probs: np.ndarray) -> float:
    p = probs + 1e-12
    return float(-np.sum(p * np.log(p)))


def features_for_modality(sample: dict, modality: str) -> Tuple[np.ndarray, int, float, bool]:
    """
    Returns (feature[6], pred_idx, confidence, available)
    feature = [max_prob, entropy, logit_mean, logit_var, is_missing, agreement]
    agreement is computed against other available modalities on 4-class prediction.
    """
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

    # agreement with other available modalities
    others = []
    for m2 in MODALITIES:
        if m2 == modality:
            continue
        o = mods.get(m2, {"available": False})
        if o.get("available", False):
            others.append(int(o["pred_idx"]))
    if len(others) == 0:
        agreement = 0.5
    else:
        agreement = float(sum(1 for p in others if p == pred) / len(others))

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
        return self.net(x)  # logits


class MRE(nn.Module):
    """Modality Reliability Estimator with per-modality small MLPs."""
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
        return out  # values in (0,1)


def mre_bce_balanced(reliabilities: Dict[str, torch.Tensor], 
                     correctness: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for m in MODALITIES:
        if (m in reliabilities) and (m in correctness):
            r = reliabilities[m].view(-1)
            y = correctness[m].view(-1)
            # balance weights by current batch pos rate
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
    """
    Simple reliability-driven decision: if unanimous -> accept; else pick highest r^m
    Returns (final_pred, decision_path)
    """
    preds = {}
    for m in MODALITIES:
        info = sample["modalities"].get(m, {"available": False})
        if info.get("available", False):
            preds[m] = int(info["pred_idx"])
    if len(preds) == 0:
        return 0, "no_modality"
    # unanimous?
    if len(set(preds.values())) == 1:
        p = list(preds.values())[0]
        return p, "unanimous"
    # choose the pred of modality with highest reliability
    best_m = None
    best_r = -1.0
    for m, p in preds.items():
        r = float(reliabilities.get(m, 0.0))
        if r > best_r:
            best_r = r
            best_m = m
    return preds[best_m], f"pick_{best_m}_by_r={best_r:.3f}"


def macro_f1(y_true: List[int], y_pred: List[int], n_classes: int = NUM_CLASSES) -> float:
    # compute per-class precision/recall/F1
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
            # conflict sample?
            preds = [int(info["pred_idx"]) for info in sample["modalities"].values() if info.get("available", False)]
            if len(preds) >= 2 and len(set(preds)) > 1:
                conflict_total += 1
                if final_pred == int(sample["label"]):
                    conflict_correct += 1
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred, NUM_CLASSES)
    conflict_acc = conflict_correct / conflict_total if conflict_total > 0 else 0.0
    return {"accuracy": acc, "macro_f1": mf1, "conflict_accuracy": conflict_acc}


def evaluate_with_records(dataset: Dataset, mre: MRE, device: torch.device):
    """
    Evaluate MRE and also return per-sample records for explainability exports.
    Each record contains: idx, label, final_pred, decision_path, conflict flag,
    per-modality predicted label and reliability.
    """
    mre.eval()
    y_true, y_pred = [], []
    conflict_correct = 0
    conflict_total = 0
    records = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            feats_t = {}
            for m in MODALITIES:
                f, _, _, avail = features_for_modality(sample, m)
                if avail:
                    feats_t[m] = torch.from_numpy(f).float().unsqueeze(0).to(device)
            rel = {}
            if feats_t:
                r = mre(feats_t)
                rel = {k: float(v.item()) for k, v in r.items()}
            final_pred, path = fuse_with_reliability(sample, rel)
            label = int(sample["label"])
            y_true.append(label)
            y_pred.append(int(final_pred))
            # conflict detection and per-modality predictions
            preds_map = {m: -1 for m in MODALITIES}
            for m in MODALITIES:
                info = sample["modalities"].get(m, {"available": False})
                if info.get("available", False):
                    preds_map[m] = int(info["pred_idx"])            
            preds_list = [p for p in preds_map.values() if p != -1]
            is_conflict = 1 if (len(preds_list) >= 2 and len(set(preds_list)) > 1) else 0
            if is_conflict:
                conflict_total += 1
                if final_pred == label:
                    conflict_correct += 1
            record = {
                "idx": i,
                "label": label,
                "final_pred": int(final_pred),
                "decision_path": path,
                "conflict": is_conflict,
                "pred_text": preds_map["text"],
                "pred_visual": preds_map["visual"],
                "pred_audio": preds_map["audio"],
                "rel_text": float(rel.get("text", 0.0)),
                "rel_visual": float(rel.get("visual", 0.0)),
                "rel_audio": float(rel.get("audio", 0.0)),
            }
            records.append(record)
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred, NUM_CLASSES)
    conflict_acc = (conflict_correct / conflict_total) if conflict_total > 0 else 0.0
    return {"accuracy": acc, "macro_f1": mf1, "conflict_accuracy": conflict_acc}, records


def evaluate_confidence_baseline_with_records(dataset: Dataset):
    """
    Baseline with confidence-as-reliability; returns metrics and per-sample records.
    """
    y_true, y_pred = [], []
    conflict_correct = 0
    conflict_total = 0
    records = []
    for i in range(len(dataset)):
        sample = dataset[i]
        # build reliability by confidence
        rel = {}
        preds_map = {m: -1 for m in MODALITIES}
        for m in MODALITIES:
            info = sample["modalities"].get(m, {"available": False})
            if info.get("available", False):
                rel[m] = float(info.get("confidence", 0.0))
                preds_map[m] = int(info["pred_idx"])
        final_pred, path = fuse_with_reliability(sample, rel)
        label = int(sample["label"])
        y_true.append(label)
        y_pred.append(int(final_pred))
        preds_list = [p for p in preds_map.values() if p != -1]
        is_conflict = 1 if (len(preds_list) >= 2 and len(set(preds_list)) > 1) else 0
        if is_conflict:
            conflict_total += 1
            if final_pred == label:
                conflict_correct += 1
        record = {
            "idx": i,
            "label": label,
            "final_pred": int(final_pred),
            "decision_path": path,
            "conflict": is_conflict,
            "pred_text": preds_map["text"],
            "pred_visual": preds_map["visual"],
            "pred_audio": preds_map["audio"],
            "rel_text": float(rel.get("text", 0.0)),
            "rel_visual": float(rel.get("visual", 0.0)),
            "rel_audio": float(rel.get("audio", 0.0)),
        }
        records.append(record)
    acc = float(np.mean([1 if a == b else 0 for a, b in zip(y_true, y_pred)]))
    mf1 = macro_f1(y_true, y_pred, NUM_CLASSES)
    conflict_acc = (conflict_correct / conflict_total) if conflict_total > 0 else 0.0
    return {"accuracy": acc, "macro_f1": mf1, "conflict_accuracy": conflict_acc}, records


def _save_records_csv(out_dir: str, split: str, protocol: str, records: List[dict], topk_conflicts: int = 50):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    # full records
    path_full = os.path.join(out_dir, f"{split}_{protocol}_records.csv")
    headers = [
        "idx","label","final_pred","decision_path","conflict",
        "pred_text","pred_visual","pred_audio",
        "rel_text","rel_visual","rel_audio"
    ]
    with open(path_full, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in records:
            row = [
                str(r["idx"]), str(r["label"]), str(r["final_pred"]), r["decision_path"], str(r["conflict"]),
                str(r["pred_text"]), str(r["pred_visual"]), str(r["pred_audio"]),
                f"{r['rel_text']:.6f}", f"{r['rel_visual']:.6f}", f"{r['rel_audio']:.6f}"
            ]
            f.write(",".join(row) + "\n")
    # conflict top-k
    conflicts = [r for r in records if int(r.get("conflict", 0)) == 1]
    path_conf = os.path.join(out_dir, f"{split}_{protocol}_conflicts_topk.csv")
    with open(path_conf, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in conflicts[:topk_conflicts]:
            row = [
                str(r["idx"]), str(r["label"]), str(r["final_pred"]), r["decision_path"], str(r["conflict"]),
                str(r["pred_text"]), str(r["pred_visual"]), str(r["pred_audio"]),
                f"{r['rel_text']:.6f}", f"{r['rel_visual']:.6f}", f"{r['rel_audio']:.6f}"
            ]
            f.write(",".join(row) + "\n")

def evaluate_confidence_baseline(dataset: Dataset) -> Dict[str, float]:
    """
    Baseline evaluation using per-modality confidence as reliability (no training, no MRE).
    Decision rule identical to fuse_with_reliability.
    """
    y_true, y_pred = [], []
    conflict_correct = 0
    conflict_total = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        # build reliability by confidence
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
    # JSON
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    # CSV for val curve
    if "epochs" in logs and logs["epochs"]:
        csv_path = os.path.join(out_dir, "val_curve.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_mre_loss,val_acc,val_macro_f1,val_conflict_acc\n")
            for e in logs["epochs"]:
                f.write(f"{e['epoch']},{e['train_mre_loss']:.6f},{e['val']['accuracy']:.6f},{e['val']['macro_f1']:.6f},{e['val']['conflict_accuracy']:.6f}\n")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # dataset split
    full = SyntheticMultimodalDataset(n_samples=args.samples, missing_rate=args.missing_rate, seed=args.seed)
    # simple split 70/15/15
    n = len(full)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    idxs = list(range(n))
    random.shuffle(idxs)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    subset = lambda idlist: torch.utils.data.Subset(full, idlist)
    train_set, val_set, test_set = subset(train_idx), subset(val_idx), subset(test_idx)

    mre = MRE(in_dim=6, h_dim=8).to(device)
    optim_mre = optim.AdamW(mre.parameters(), lr=args.lr)

    # prepare logs and output dir
    out_dir = args.out_dir
    logs = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "args": vars(args),
        "baseline": {},
        "epochs": []
    }

    # Baseline (confidence-as-reliability) before training
    val_conf_base = evaluate_confidence_baseline(val_set)
    test_conf_base = evaluate_confidence_baseline(test_set)
    print("Baseline (confidence-as-reliability):")
    print(f"- Val  | Acc: {val_conf_base['accuracy']:.4f} | MF1: {val_conf_base['macro_f1']:.4f} | ConflictAcc: {val_conf_base['conflict_accuracy']:.4f}")
    print(f"- Test | Acc: {test_conf_base['accuracy']:.4f} | MF1: {test_conf_base['macro_f1']:.4f} | ConflictAcc: {test_conf_base['conflict_accuracy']:.4f}")
    logs["baseline"] = {"val": val_conf_base, "test": test_conf_base}
    _maybe_save_logs(out_dir, logs)

    # Stage A: no encoder training here (synthetic predictors are fixed)
    print("Stage A (synthetic): no encoder training of unimodal encoders; proceed to Stage B (MRE)")

    # Stage B: train MRE with weak supervision
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    for epoch in range(1, args.epochs + 1):
        mre.train()
        running = []
        for batch in loader:
            # batch is a list of dicts unless collate; process manually
            optim_mre.zero_grad()
            feats_t = {"text": [], "visual": [], "audio": []}
            corr_t = {"text": [], "visual": [], "audio": []}
            any_available = {"text": False, "visual": False, "audio": False}
            for sample in batch:
                label = int(sample["label"])  # scalar
                # first compute all preds for agreement measure
                tmp_preds = {}
                for m in MODALITIES:
                    f, pred, conf, avail = features_for_modality(sample, m)
                    if avail:
                        tmp_preds[m] = pred
                # then recompute features with agreement using tmp_preds (already handled in features_for_modality)
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
        # epoch logs
        train_loss = float(np.mean(running)) if running else 0.0
        val_metrics = evaluate(val_set, mre, device)
        print(f"Epoch {epoch:02d} | L_mre: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val MF1: {val_metrics['macro_f1']:.4f} | Val ConflictAcc: {val_metrics['conflict_accuracy']:.4f}")
        logs["epochs"].append({
            "epoch": epoch,
            "train_mre_loss": train_loss,
            "val": val_metrics
        })
        _maybe_save_logs(out_dir, logs)

    # final eval: MRE vs confidence baseline
    test_mre = evaluate(test_set, mre, device)
    print("\nFinal Test Metrics:")
    print(f"- Confidence Baseline | Acc: {test_conf_base['accuracy']:.4f} | MF1: {test_conf_base['macro_f1']:.4f} | ConflictAcc: {test_conf_base['conflict_accuracy']:.4f}")
    print(f"- MRE (weakly-supervised) | Acc: {test_mre['accuracy']:.4f} | MF1: {test_mre['macro_f1']:.4f} | ConflictAcc: {test_mre['conflict_accuracy']:.4f}")
    logs["test_mre"] = test_mre
    _maybe_save_logs(out_dir, logs)
    # explainability exports (test split)
    mre_metrics2, mre_records = evaluate_with_records(test_set, mre, device)
    base_metrics2, base_records = evaluate_confidence_baseline_with_records(test_set)
    _save_records_csv(out_dir, "test", "mre", mre_records)
    _save_records_csv(out_dir, "test", "baseline", base_records)

    # return summary for possible aggregation
    run_summary = {
        "seed": args.seed,
        "val_baseline": val_conf_base,
        "test_baseline": test_conf_base,
        "test_mre": test_mre
    }
    return run_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_mre", type=float, default=0.3)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seeds", type=str, default="", help="comma or space separated list of seeds for multi-run aggregation")
    args = parser.parse_args()
    # multi-seed mode
    seeds_str = (args.seeds or "").strip()
    if seeds_str:
        # parse seeds list
        parts = [p for p in seeds_str.replace(",", " ").split(" ") if p]
        seeds = [int(p) for p in parts]
        root_out = args.out_dir
        agg = {
            "seeds": seeds,
            "test_baseline": [],
            "test_mre": []
        }
        for s in seeds:
            local_args = argparse.Namespace(**vars(args))
            local_args.seed = s
            local_args.out_dir = os.path.join(root_out, f"seed_{s}") if root_out else ""
            set_seed(local_args.seed)
            summary = train(local_args)
            agg["test_baseline"].append(summary["test_baseline"])
            agg["test_mre"].append(summary["test_mre"])
        def _mean_std(items, key):
            vals = [float(x[key]) for x in items]
            m = float(statistics.mean(vals)) if len(vals) > 0 else 0.0
            try:
                s = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
            except Exception:
                s = 0.0
            return {"mean": m, "std": s}
        aggregate = {
            "seeds": seeds,
            "test_baseline": {
                "accuracy": _mean_std(agg["test_baseline"], "accuracy"),
                "macro_f1": _mean_std(agg["test_baseline"], "macro_f1"),
                "conflict_accuracy": _mean_std(agg["test_baseline"], "conflict_accuracy"),
            },
            "test_mre": {
                "accuracy": _mean_std(agg["test_mre"], "accuracy"),
                "macro_f1": _mean_std(agg["test_mre"], "macro_f1"),
                "conflict_accuracy": _mean_std(agg["test_mre"], "conflict_accuracy"),
            }
        }
        print("\nAggregate over seeds:")
        for proto in ["test_baseline", "test_mre"]:
            am = aggregate[proto]
            print(f"- {proto}: Acc {am['accuracy']['mean']:.4f}±{am['accuracy']['std']:.4f} | MF1 {am['macro_f1']['mean']:.4f}±{am['macro_f1']['std']:.4f} | ConflictAcc {am['conflict_accuracy']['mean']:.4f}±{am['conflict_accuracy']['std']:.4f}")
        # save aggregate
        if root_out:
            os.makedirs(root_out, exist_ok=True)
            with open(os.path.join(root_out, "aggregate.json"), "w", encoding="utf-8") as f:
                json.dump(aggregate, f, ensure_ascii=False, indent=2)
    else:
        set_seed(args.seed)
        train(args)
