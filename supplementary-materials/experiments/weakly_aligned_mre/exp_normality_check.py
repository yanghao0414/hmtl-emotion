# -*- coding: utf-8 -*-
"""
Shapiro-Wilk正态性检验：验证paired t-test的前提假设
对significance_test_results.csv中的per-seed差值进行正态性检验
"""
import csv
from scipy import stats

csv_path = "outputs/significance_test/significance_test_results.csv"

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=" * 70)
print("Shapiro-Wilk正态性检验（paired differences）")
print("=" * 70)

for row in rows:
    cond = row["condition"]
    mre_vals = [float(x) for x in row["per_seed_mre"].split(";")]
    wt_vals = [float(x) for x in row["per_seed_wt"].split(";")]
    diffs = [m - w for m, w in zip(mre_vals, wt_vals)]
    
    stat, p = stats.shapiro(diffs)
    normal = "Yes" if p > 0.05 else "No"
    print(f"\n{cond}:")
    print(f"  diffs = {[f'{d:.4f}' for d in diffs]}")
    print(f"  Shapiro-Wilk W={stat:.4f}, p={p:.4f}  -> Normal: {normal}")

print("\n" + "=" * 70)
print("结论：若所有p > 0.05，则满足正态性假设，paired t-test有效。")
print("若不满足，Wilcoxon检验（已报告）作为非参数替代同样有效。")
print("=" * 70)
