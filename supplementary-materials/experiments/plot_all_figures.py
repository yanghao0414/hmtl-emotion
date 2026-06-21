# -*- coding: utf-8 -*-
"""
论文全部图表绘制脚本 — 学术风格（黑白友好、高DPI、LaTeX-ready）

生成7张图，保存为PNG和PDF：
  图1  消融实验分组柱状图
  图2  文本噪声干扰折线图（核心图）
  图3  模态缺失率折线图
  图4a λ_ent敏感性折线图
  图4b 学习率敏感性折线图
  图5  多种子稳定性箱线图
  图6  置信度稳定性双Y轴折线图
  图7  小样本学习曲线折线图

Run:
  python plot_all_figures.py
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm

# ── 全局样式 ──
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["cn", "en"], default="cn")
    p.add_argument("--out_dir", type=str, default="")
    return p.parse_args()

args = _parse_args()

# 尝试加载中文字体
zh_font = None
for fname in ['SimHei', 'Microsoft YaHei', 'STHeiti', 'PingFang SC', 'Source Han Sans CN']:
    try:
        fp = fm.findfont(fm.FontProperties(family=fname), fallback_to_default=False)
        if fp and 'LastResort' not in fp:
            zh_font = fm.FontProperties(fname=fp)
            break
    except Exception:
        continue

if args.lang == "en":
    USE_CN = False
else:
    if zh_font is None:
        print("Warning: 未找到中文字体，图表标签将使用英文")
        USE_CN = False
    else:
        USE_CN = True
        plt.rcParams['font.family'] = zh_font.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        print(f"使用中文字体: {zh_font.get_name()}")

OUT_DIR = args.out_dir or ("figures" if USE_CN else "figures_en")
os.makedirs(OUT_DIR, exist_ok=True)

# 配色方案 — 学术风格，对色盲友好
C_MRE    = '#2166AC'  # 蓝
C_WV     = '#B2182B'  # 红
C_MV     = '#666666'  # 灰
C_CONF   = '#4DAF4A'  # 绿
C_TEXT   = '#E69F00'  # 橙
C_VISUAL = '#56B4E9'  # 浅蓝
C_AUDIO  = '#009E73'  # 青绿

MARKERS = {'MRE': 'o', 'WV': 's', 'MV': '^', 'CONF': 'D'}

def save(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
    print(f"  已保存: {name}.png / .pdf")
    plt.close(fig)

# ================================================================
# 图1：消融实验分组柱状图
# ================================================================
def plot_fig1():
    print("\n绘制 图1 消融实验...")
    configs = ['T', 'V', 'A', 'T+V', 'T+A', 'V+A', 'T+V+A']
    mre  = [78.53, 63.11, 74.89, 80.40, 78.22, 75.73, 85.82]
    wv   = [None,  None,  None,  80.49, 78.98, 76.00, 85.78]
    mv   = [None,  None,  None,  70.67, 76.36, 68.22, 82.98]
    conf = [None,  None,  None,  70.67, 76.36, 68.22, 73.56]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(configs))
    w = 0.18

    # 单模态只画一个bar，多模态画4个
    for i in range(3):
        ax.bar(x[i], mre[i], w, color=C_MRE, edgecolor='white', linewidth=0.5)
        ax.text(x[i], mre[i] + 0.5, f'{mre[i]:.1f}', ha='center', va='bottom', fontsize=8)

    for i in range(3, 7):
        bars_data = [
            (mre[i],  C_MRE,  'MRE'),
            (wv[i],   C_WV,   'Weighted Voting' if not USE_CN else '加权投票'),
            (mv[i],   C_MV,   'Majority Voting' if not USE_CN else '多数投票'),
            (conf[i], C_CONF, 'Confidence' if not USE_CN else '置信度融合'),
        ]
        for j, (val, color, _) in enumerate(bars_data):
            if val is not None:
                offset = (j - 1.5) * w
                bar = ax.bar(x[i] + offset, val, w, color=color, edgecolor='white', linewidth=0.5)
                ax.text(x[i] + offset, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_MRE,  label='MRE'),
        Patch(facecolor=C_WV,   label='加权投票' if USE_CN else 'Weighted Voting'),
        Patch(facecolor=C_MV,   label='多数投票' if USE_CN else 'Majority Voting'),
        Patch(facecolor=C_CONF, label='置信度融合' if USE_CN else 'Confidence'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('模态配置' if USE_CN else 'Modality Configuration')
    ax.set_ylim(55, 92)
    ax.set_title('图1  消融实验：模态配置与融合方法对比' if USE_CN else
                 'Fig.1  Ablation: Modality Configuration Comparison')
    fig.tight_layout()
    save(fig, 'fig1_ablation')


# ================================================================
# 图2：文本噪声干扰折线图（核心图）
# ================================================================
def plot_fig2():
    print("绘制 图2 文本噪声干扰...")
    noise = [0, 5, 10, 15, 20, 30, 40, 50]
    mre     = [85.82, 82.62, 80.44, 79.73, 78.49, 78.44, 77.69, 75.64]
    mre_sd  = [1.04,  1.09,  1.02,  0.69,  0.56,  1.63,  1.34,  0.20]
    wv      = [85.78, 82.09, 80.67, 78.67, 75.78, 73.33, 69.20, 65.47]
    mv      = [84.22, 81.20, 79.38, 77.11, 74.89, 72.40, 68.36, 64.62]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(noise,
                    [m - s for m, s in zip(mre, mre_sd)],
                    [m + s for m, s in zip(mre, mre_sd)],
                    alpha=0.15, color=C_MRE)
    ax.plot(noise, mre, color=C_MRE, marker='o', label='MRE', zorder=5)
    ax.plot(noise, wv,  color=C_WV,  marker='s', label='加权投票' if USE_CN else 'Weighted Voting')
    ax.plot(noise, mv,  color=C_MV,  marker='^', label='多数投票' if USE_CN else 'Majority Voting')

    # 标注50%处差距
    ax.annotate('',
                xy=(50, 75.64), xytext=(50, 65.47),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
    ax.text(51.5, 70.5, '+10.17pp', fontsize=10, fontweight='bold', color='#333333')

    ax.set_xlabel('文本噪声强度 (%)' if USE_CN else 'Text Noise Level (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(60, 90)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_title('图2  文本噪声干扰下的融合性能' if USE_CN else
                 'Fig.2  Fusion Performance under Text Noise')
    fig.tight_layout()
    save(fig, 'fig2_text_noise')


# ================================================================
# 图3：模态缺失率折线图
# ================================================================
def plot_fig3():
    print("绘制 图3 模态缺失率...")
    miss    = [0, 5, 10, 20, 30, 50]
    mre     = [86.22, 84.76, 85.82, 79.47, 78.31, 71.87]
    mre_sd  = [1.21,  0.73,  1.04,  2.27,  0.73,  1.41]
    wv      = [86.80, 86.84, 85.78, 82.09, 78.67, 72.27]
    wv_sd   = [0.96,  1.07,  0.89,  2.28,  1.06,  1.62]
    conf    = [72.71, 74.93, 73.56, 72.09, 71.60, 68.04]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(miss,
                    [m - s for m, s in zip(mre, mre_sd)],
                    [m + s for m, s in zip(mre, mre_sd)],
                    alpha=0.15, color=C_MRE)
    ax.fill_between(miss,
                    [m - s for m, s in zip(wv, wv_sd)],
                    [m + s for m, s in zip(wv, wv_sd)],
                    alpha=0.10, color=C_WV)
    ax.plot(miss, mre,  color=C_MRE,  marker='o', label='MRE')
    ax.plot(miss, wv,   color=C_WV,   marker='s', label='加权投票' if USE_CN else 'Weighted Voting')
    ax.plot(miss, conf, color=C_CONF, marker='D', label='置信度融合' if USE_CN else 'Confidence', linestyle='--')

    ax.set_xlabel('模态缺失率 (%)' if USE_CN else 'Modality Missing Rate (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(65, 92)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_title('图3  模态缺失率对融合性能的影响' if USE_CN else
                 'Fig.3  Impact of Modality Missing Rate')
    fig.tight_layout()
    save(fig, 'fig3_missing_rate')


# ================================================================
# 图4a：λ_ent 敏感性
# ================================================================
def plot_fig4a():
    print("绘制 图4a λ_ent敏感性...")
    lam   = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00, 5.00, 10.00]
    acc   = [82.45, 82.32, 82.37, 82.43, 82.37, 82.48, 82.48, 82.00, 80.45, 80.67, 80.00]
    sd    = [1.33,  1.44,  1.39,  1.44,  1.46,  1.50,  1.76,  1.78,  2.23,  2.67,  2.51]

    fig, ax = plt.subplots(figsize=(7, 4))

    log_lam = [np.log10(l) for l in lam]
    ax.fill_between(log_lam,
                    [a - s for a, s in zip(acc, sd)],
                    [a + s for a, s in zip(acc, sd)],
                    alpha=0.2, color=C_MRE)
    ax.plot(log_lam, acc, color=C_MRE, marker='o')

    # 标注稳定区域
    ax.axvspan(np.log10(0.001), np.log10(0.5), alpha=0.06, color='green')
    ax.text(np.log10(0.03), 83.3,
            '稳定区间 (82.3%~82.5%)' if USE_CN else 'Stable range',
            fontsize=9, color='green', ha='center')

    ax.set_xlabel(r'$\lambda_{ent}$  (log scale)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(log_lam)
    ax.set_xticklabels([str(l) for l in lam], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(76, 85)
    ax.set_title(r'图4a  $\lambda_{ent}$ 敏感性分析' if USE_CN else
                 r'Fig.4a  $\lambda_{ent}$ Sensitivity')
    fig.tight_layout()
    save(fig, 'fig4a_lambda_sensitivity')


# ================================================================
# 图4b：学习率敏感性
# ================================================================
def plot_fig4b():
    print("绘制 图4b 学习率敏感性...")
    lr  = [1e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2]
    acc = [78.62, 83.24, 83.38, 85.82, 85.20, 85.02, 85.78]
    sd  = [3.86,  1.79,  1.89,  1.04,  1.22,  1.60,  0.60]

    fig, ax = plt.subplots(figsize=(7, 4))

    log_lr = [np.log10(l) for l in lr]
    ax.fill_between(log_lr,
                    [a - s for a, s in zip(acc, sd)],
                    [a + s for a, s in zip(acc, sd)],
                    alpha=0.2, color=C_MRE)
    ax.plot(log_lr, acc, color=C_MRE, marker='o')

    # 标注lr=1e-4异常低
    ax.annotate(f'78.6%\n(lr过小)' if USE_CN else '78.6%\n(lr too small)',
                xy=(log_lr[0], acc[0]),
                xytext=(log_lr[0] + 0.3, acc[0] - 2),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(log_lr)
    ax.set_xticklabels([f'{l:.0e}' for l in lr], rotation=45, ha='right', fontsize=8)
    ax.set_ylim(72, 90)
    ax.set_title('图4b  学习率敏感性分析' if USE_CN else
                 'Fig.4b  Learning Rate Sensitivity')
    fig.tight_layout()
    save(fig, 'fig4b_lr_sensitivity')


# ================================================================
# 图5：多种子稳定性箱线图
# ================================================================
def plot_fig5():
    print("绘制 图5 多种子稳定性...")
    mre_vals = [82.40, 86.67, 84.67, 83.87, 84.93, 84.13, 82.00, 86.13, 85.20, 83.47]
    wv_vals  = [85.47, 86.53, 84.80, 83.47, 84.80, 84.27, 82.27, 86.00, 85.07, 83.33]
    mv_vals  = [85.07, 84.93, 83.07, 83.07, 83.87, 83.87, 80.00, 84.67, 84.27, 81.07]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    data = [mre_vals, wv_vals, mv_vals]
    labels = ['MRE', '加权投票' if USE_CN else 'Weighted\nVoting',
              '多数投票' if USE_CN else 'Majority\nVoting']
    colors = [C_MRE, C_WV, C_MV]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # 叠加散点
    for i, (vals, color) in enumerate(zip(data, colors)):
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([i + 1 + j for j in jitter], vals, color=color,
                   alpha=0.7, s=25, zorder=5, edgecolors='white', linewidths=0.5)

    # 标注均值和SD
    for i, vals in enumerate(data):
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ax.text(i + 1, 79.5, f'μ={mean:.1f}%\nσ={std:.1f}%',
                ha='center', fontsize=8, color='#333')

    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(78, 88)
    ax.set_title('图5  多种子稳定性 (n=10)' if USE_CN else
                 'Fig.5  Multi-seed Stability (n=10)')
    fig.tight_layout()
    save(fig, 'fig5_multi_seed')


# ================================================================
# 图6：置信度稳定性双Y轴折线图
# ================================================================
def plot_fig6():
    print("绘制 图6 置信度稳定性...")
    noise     = [0, 10, 20, 30, 50]
    acc       = [85.82, 78.31, 69.20, 60.58, 43.82]
    rel_text  = [0.6505, 0.5906, 0.5562, 0.5326, 0.4928]
    rel_vis   = [0.5978, 0.5416, 0.5129, 0.4944, 0.4926]
    rel_aud   = [0.6570, 0.5806, 0.5381, 0.5095, 0.4991]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # 左Y轴：准确率
    ax1.plot(noise, acc, color=C_MRE, marker='o', linewidth=2.2,
             label='Fusion Accuracy', zorder=5)
    ax1.set_ylabel('Accuracy (%)', color=C_MRE)
    ax1.tick_params(axis='y', labelcolor=C_MRE)
    ax1.set_ylim(35, 92)

    # 右Y轴：可靠性评分
    ax2.plot(noise, rel_text, color=C_TEXT,   marker='s', linestyle='--',
             label='文本可靠性' if USE_CN else 'Text Rel.', alpha=0.85)
    ax2.plot(noise, rel_vis,  color=C_VISUAL, marker='^', linestyle='--',
             label='视觉可靠性' if USE_CN else 'Visual Rel.', alpha=0.85)
    ax2.plot(noise, rel_aud,  color=C_AUDIO,  marker='D', linestyle='--',
             label='音频可靠性' if USE_CN else 'Audio Rel.', alpha=0.85)

    # 标注0.5参考线
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.text(52, 0.505, '0.5 (不确定)' if USE_CN else '0.5 (uncertain)',
             fontsize=8, color='gray')

    ax2.set_ylabel('Reliability Score', color='#555')
    ax2.tick_params(axis='y', labelcolor='#555')
    ax2.set_ylim(0.40, 0.72)

    ax1.set_xlabel('全模态噪声强度 (%)' if USE_CN else 'All-modality Noise Level (%)')

    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8.5,
               framealpha=0.9)

    ax1.set_title('图6  噪声下MRE可靠性评分变化' if USE_CN else
                  'Fig.6  MRE Reliability Scores under Noise')

    # 右侧spine需要显示
    ax1.spines['right'].set_visible(True)
    fig.tight_layout()
    save(fig, 'fig6_confidence_stability')


# ================================================================
# 图7：小样本学习曲线
# ================================================================
def plot_fig7():
    print("绘制 图7 小样本学习曲线...")
    ratio   = [5, 10, 20, 50, 70, 100]
    mre     = [78.98, 82.22, 82.76, 83.60, 84.36, 85.11]
    mre_sd  = [4.82,  2.02,  1.70,  1.96,  2.07,  1.00]
    wv      = [85.78, 85.78, 85.78, 85.78, 85.78, 85.78]
    mv      = [84.22, 84.22, 84.22, 84.22, 84.22, 84.22]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(ratio,
                    [m - s for m, s in zip(mre, mre_sd)],
                    [m + s for m, s in zip(mre, mre_sd)],
                    alpha=0.15, color=C_MRE)
    ax.plot(ratio, mre, color=C_MRE, marker='o', label='MRE')
    ax.plot(ratio, wv,  color=C_WV,  marker='s', linestyle='--',
            label='加权投票' if USE_CN else 'Weighted Voting')
    ax.plot(ratio, mv,  color=C_MV,  marker='^', linestyle='--',
            label='多数投票' if USE_CN else 'Majority Voting')

    # 标注交叉点附近
    ax.annotate('100%时接近加权投票' if USE_CN else 'Converges at 100%',
                xy=(100, 85.11), xytext=(70, 86.5),
                fontsize=8.5, color=C_MRE,
                arrowprops=dict(arrowstyle='->', color=C_MRE, lw=1))

    ax.set_xlabel('训练数据比例 (%)' if USE_CN else 'Training Data Ratio (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(72, 90)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_title('图7  小样本学习曲线' if USE_CN else
                 'Fig.7  Few-shot Learning Curve')
    fig.tight_layout()
    save(fig, 'fig7_sample_efficiency')


# ================================================================
# Main
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("开始绘制论文全部图表...")
    print(f"输出目录: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)

    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4a()
    plot_fig4b()
    plot_fig5()
    plot_fig6()
    plot_fig7()

    print("\n" + "=" * 60)
    print(f"全部7张图已保存至 {os.path.abspath(OUT_DIR)}/")
    print("每张图均有 PNG (300dpi) 和 PDF 两个版本")
    print("=" * 60)
