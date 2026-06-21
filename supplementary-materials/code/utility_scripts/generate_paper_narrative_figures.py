#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成论文叙述配套图片
针对青少年情绪识别研究的专用图表
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r"d:\bigcreate\10_论文资料\青少年情绪识别研究"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 青少年心理健康现状图
# ============================================================
def generate_mental_health_status():
    """青少年心理健康问题统计"""
    print("生成青少年心理健康现状图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    problems = ['学业压力', '情绪波动', '焦虑症状', '社交困难', '抑郁倾向']
    rates = [72.8, 45.2, 31.3, 28.5, 24.6]
    colors = ['#9b59b6', '#f39c12', '#e74c3c', '#1abc9c', '#3498db']
    
    bars = ax.barh(problems, rates, color=colors, height=0.6)
    
    # 添加数值标签
    for bar, rate in zip(bars, rates):
        ax.text(rate + 1.5, bar.get_y() + bar.get_height()/2, 
               f'{rate}%', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('发生率 (%)', fontsize=12)
    ax.set_xlim(0, 85)
    ax.set_title('中国青少年心理健康问题发生率', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数据来源
    ax.text(0.98, 0.02, '数据来源: 中科院心理所《国民心理健康报告》', 
           transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '01_青少年心理健康现状.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 2. 青少年情绪特点对比图
# ============================================================
def generate_emotion_characteristics():
    """青少年vs成人情绪特点对比"""
    print("生成青少年情绪特点对比图...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    characteristics = ['情绪强度', '变化速度', '混合情绪', '隐蔽性', '环境敏感']
    adolescent = [0.85, 0.90, 0.82, 0.75, 0.88]
    adult = [0.60, 0.55, 0.50, 0.45, 0.55]
    
    x = np.arange(len(characteristics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, adolescent, width, label='青少年', color='#e74c3c')
    bars2 = ax.bar(x + width/2, adult, width, label='成人', color='#3498db')
    
    ax.set_ylabel('相对程度', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(characteristics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title('青少年 vs 成人情绪表达特点对比', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加注释
    ax.annotate('青少年情绪\n波动更剧烈', xy=(1, 0.90), xytext=(1.8, 1.0),
               arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
               fontsize=10, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '02_青少年情绪特点对比.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 3. Russell情绪环形模型与4类映射
# ============================================================
def generate_russell_4class_mapping():
    """Russell模型与4类情绪映射"""
    print("生成Russell模型与4类映射图...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制坐标轴
    ax.axhline(y=0, color='gray', linewidth=1.5, zorder=1)
    ax.axvline(x=0, color='gray', linewidth=1.5, zorder=1)
    
    # 绘制圆形
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, zorder=2)
    
    # 4类区域填充
    # 积极 (右半部分)
    theta_pos = np.linspace(-np.pi/2, np.pi/2, 50)
    ax.fill(np.append(np.cos(theta_pos), 0), np.append(np.sin(theta_pos), 0), 
           alpha=0.2, color='#2ecc71', zorder=0)
    
    # 激活消极 (左上)
    theta_act_neg = np.linspace(np.pi/2, np.pi, 25)
    ax.fill(np.append(np.cos(theta_act_neg), [0, 0]), 
           np.append(np.sin(theta_act_neg), [0, 1]), 
           alpha=0.2, color='#e74c3c', zorder=0)
    
    # 非激活消极 (左下)
    theta_deact_neg = np.linspace(np.pi, 3*np.pi/2, 25)
    ax.fill(np.append(np.cos(theta_deact_neg), [0, 0]), 
           np.append(np.sin(theta_deact_neg), [-1, 0]), 
           alpha=0.2, color='#3498db', zorder=0)
    
    # 平静 (右下部分 - 简化处理)
    
    # 7类情绪位置
    emotions_7 = {
        '愤怒': (-0.7, 0.75, '#e74c3c'),
        '焦虑': (-0.5, 0.6, '#e67e22'),
        '快乐': (0.75, 0.6, '#2ecc71'),
        '悲伤': (-0.6, -0.5, '#3498db'),
        '失望': (-0.4, -0.35, '#9b59b6'),
        '支持': (0.6, 0.35, '#1abc9c'),
        '平静': (0.3, -0.3, '#95a5a6'),
    }
    
    for emotion, (v, a, color) in emotions_7.items():
        ax.scatter(v, a, s=250, c=color, edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(emotion, (v, a), xytext=(8, 8), textcoords='offset points',
                   fontsize=11, fontweight='bold')
    
    # 4类标签
    ax.text(0.5, 0.7, '积极', ha='center', va='center', fontsize=16, 
           fontweight='bold', color='#27ae60',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-0.6, 0.75, '激活消极', ha='center', va='center', fontsize=16,
           fontweight='bold', color='#c0392b',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-0.6, -0.6, '非激活消极', ha='center', va='center', fontsize=16,
           fontweight='bold', color='#2980b9',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.5, -0.5, '平静', ha='center', va='center', fontsize=16,
           fontweight='bold', color='#7f8c8d',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 坐标轴标签
    ax.text(1.15, 0, 'Valence +\n(愉悦)', ha='left', va='center', fontsize=11)
    ax.text(-1.15, 0, 'Valence -\n(不愉悦)', ha='right', va='center', fontsize=11)
    ax.text(0, 1.15, 'Arousal + (高激活)', ha='center', va='bottom', fontsize=11)
    ax.text(0, -1.15, 'Arousal - (低激活)', ha='center', va='top', fontsize=11)
    
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title('基于Russell情绪环形模型的7类→4类映射', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '03_Russell模型4类映射.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 4. 4类情绪与青少年干预策略
# ============================================================
def generate_4class_intervention():
    """4类情绪与干预策略对应图"""
    print("生成4类情绪干预策略图...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    classes = [
        {
            'name': '积极',
            'emotions': '快乐、支持',
            'color': '#2ecc71',
            'intervention': '强化积极行为\n建立情绪资源库\n培养抗压能力',
            'urgency': '★☆☆',
            'x': 0.5
        },
        {
            'name': '激活消极',
            'emotions': '愤怒、焦虑',
            'color': '#e74c3c',
            'intervention': '即时情绪疏导\n压力管理训练\n冲突解决技巧',
            'urgency': '★★★',
            'x': 3.75
        },
        {
            'name': '非激活消极',
            'emotions': '悲伤、失望',
            'color': '#3498db',
            'intervention': '心理支持陪伴\n社交激活\n专业评估转介',
            'urgency': '★★☆',
            'x': 7
        },
        {
            'name': '平静',
            'emotions': '平静、中性',
            'color': '#95a5a6',
            'intervention': '维持日常监测\n情绪觉察训练\n预防性教育',
            'urgency': '★☆☆',
            'x': 10.25
        },
    ]
    
    for cls in classes:
        # 主框
        box = FancyBboxPatch((cls['x'], 2), 3, 5.5, boxstyle="round,pad=0.1",
                              facecolor=cls['color'], edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        
        # 类别名称
        ax.text(cls['x']+1.5, 7, cls['name'], ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')
        
        # 包含情绪
        ax.text(cls['x']+1.5, 6.3, f"({cls['emotions']})", ha='center', va='center',
               fontsize=10, color='white')
        
        # 干预紧迫度
        ax.text(cls['x']+1.5, 5.5, f"干预紧迫度: {cls['urgency']}", ha='center', va='center',
               fontsize=10, color='white')
        
        # 干预策略框
        intervention_box = FancyBboxPatch((cls['x']+0.2, 2.3), 2.6, 2.8, 
                                           boxstyle="round,pad=0.05",
                                           facecolor='white', edgecolor='black', linewidth=1)
        ax.add_patch(intervention_box)
        ax.text(cls['x']+1.5, 4.5, '干预策略', ha='center', va='center',
               fontsize=11, fontweight='bold', color='black')
        ax.text(cls['x']+1.5, 3.5, cls['intervention'], ha='center', va='center',
               fontsize=9, color='black')
    
    # 标题
    ax.text(7, 7.8, '4类情绪分类与青少年心理干预策略', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    # 说明
    ax.text(7, 0.8, '★★★ 需即时干预  ★★☆ 需持续关注  ★☆☆ 常规监测',
           ha='center', va='center', fontsize=11, style='italic', color='gray')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '04_4类情绪干预策略.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 5. HMTL层次化多任务学习架构
# ============================================================
def generate_hmtl_architecture():
    """HMTL模型架构图"""
    print("生成HMTL架构图...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    colors = {
        'input': '#3498db',
        'backbone': '#9b59b6',
        'shared': '#f39c12',
        'main': '#2ecc71',
        'aux': '#95a5a6',
    }
    
    # 输入层
    input_box = FancyBboxPatch((5, 8.5), 4, 1, boxstyle="round,pad=0.05",
                                facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 9, '输入 (Text/Audio/Visual)', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    
    # 骨干网络
    backbone_box = FancyBboxPatch((5, 6.5), 4, 1.5, boxstyle="round,pad=0.05",
                                   facecolor=colors['backbone'], edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(7, 7.5, '骨干网络', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(7, 7, 'BERT / Wav2Vec2 / EfficientNet', ha='center', va='center', fontsize=10, color='white')
    
    # 共享特征层
    shared_box = FancyBboxPatch((5, 4.8), 4, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=colors['shared'], edgecolor='black', linewidth=2)
    ax.add_patch(shared_box)
    ax.text(7, 5.4, '共享特征层', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 箭头
    ax.annotate('', xy=(7, 8), xytext=(7, 8.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7, 6), xytext=(7, 6.5), arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # 多任务输出
    tasks = [
        ('7类情绪\n(辅助)', 0.5, colors['aux'], '细粒度特征学习'),
        ('4类情绪\n(主任务)', 3.5, colors['main'], '青少年干预决策'),
        ('3类极性\n(辅助)', 6.5, colors['aux'], '情感倾向判断'),
        ('Arousal\n(辅助)', 9.5, colors['aux'], '激活度评估'),
        ('Valence\n(辅助)', 12, colors['aux'], '愉悦度评估'),
    ]
    
    for name, x, color, desc in tasks:
        # 任务框
        task_box = FancyBboxPatch((x, 2.5), 2.2, 1.8, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(task_box)
        ax.text(x+1.1, 3.7, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(x+1.1, 2.9, desc, ha='center', va='center', fontsize=8, color='white')
        
        # 连接线
        ax.annotate('', xy=(x+1.1, 4.3), xytext=(7, 4.8),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                  connectionstyle="arc3,rad=0.1"))
    
    # 突出主任务
    highlight = FancyBboxPatch((3.3, 2.3), 2.6, 2.2, boxstyle="round,pad=0.05",
                                facecolor='none', edgecolor='#27ae60', linewidth=3, linestyle='--')
    ax.add_patch(highlight)
    ax.text(4.6, 1.8, '★ 主输出', ha='center', va='center', fontsize=10, 
           fontweight='bold', color='#27ae60')
    
    # 标题
    ax.text(7, 9.7, 'HMTL: 层次化多任务学习架构', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    # 说明
    ax.text(7, 1, '设计理念: 7类细分情绪作为辅助任务帮助学习细粒度特征，4类核心情绪作为主任务直接服务于青少年心理干预',
           ha='center', va='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '05_HMTL架构图.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 6. 多模态融合策略
# ============================================================
def generate_multimodal_fusion():
    """多模态融合策略图"""
    print("生成多模态融合策略图...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 三个模态
    modalities = [
        ('文本模态', '社交媒体\n作文日记\n聊天记录', '#3498db', 1),
        ('语音模态', '课堂发言\n语音消息\n日常对话', '#e74c3c', 5.5),
        ('视觉模态', '面部表情\n课堂状态\n视频通话', '#2ecc71', 10),
    ]
    
    for name, scenes, color, x in modalities:
        # 模态框
        box = FancyBboxPatch((x, 6.5), 3, 2.5, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.5, 8.5, name, ha='center', va='center', fontsize=13, fontweight='bold', color='white')
        ax.text(x+1.5, 7.3, scenes, ha='center', va='center', fontsize=10, color='white')
        
        # HMTL处理
        hmtl_box = FancyBboxPatch((x, 4.5), 3, 1.5, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(hmtl_box)
        ax.text(x+1.5, 5.25, 'HMTL\n特征提取', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 箭头
        ax.annotate('', xy=(x+1.5, 6), xytext=(x+1.5, 6.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # 融合层
    fusion_box = FancyBboxPatch((4, 2.5), 6, 1.5, boxstyle="round,pad=0.1",
                                 facecolor='#9b59b6', edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(7, 3.5, '多模态融合层', ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    ax.text(7, 2.9, 'Attention-based Fusion', ha='center', va='center', fontsize=10, color='white')
    
    # 连接线到融合层
    for x in [2.5, 7, 11.5]:
        ax.annotate('', xy=(7, 4), xytext=(x, 4.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                  connectionstyle="arc3,rad=0.1"))
    
    # 输出
    output_box = FancyBboxPatch((4, 0.8), 6, 1.2, boxstyle="round,pad=0.1",
                                 facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 1.4, '最终输出: 4类情绪 + Arousal + Valence', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')
    
    ax.annotate('', xy=(7, 2), xytext=(7, 2.5),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # 标题
    ax.text(7, 9.5, '多模态情绪识别融合策略', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '06_多模态融合策略.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 7. 迁移学习策略说明
# ============================================================
def generate_transfer_learning():
    """迁移学习策略图"""
    print("生成迁移学习策略图...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 左侧: 通用数据集训练
    ax.text(3.5, 7.5, '阶段1: 基础训练', ha='center', va='center',
           fontsize=14, fontweight='bold')
    
    datasets = [
        ('中文情绪文本', '#3498db', 1),
        ('CNSCED语音', '#e74c3c', 3),
        ('FER2013表情', '#2ecc71', 5),
    ]
    
    for name, color, y in datasets:
        box = FancyBboxPatch((1, y), 5, 1.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(3.5, y+0.75, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 箭头
    ax.annotate('', xy=(7, 4), xytext=(6.2, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax.text(6.6, 4.5, '迁移', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 右侧: 青少年适配
    ax.text(10.5, 7.5, '阶段2: 青少年适配', ha='center', va='center',
           fontsize=14, fontweight='bold')
    
    adaptations = [
        ('4类分类体系\n(针对干预需求设计)', '#9b59b6', 5),
        ('层次化多任务学习\n(捕捉混合情绪)', '#f39c12', 3),
        ('应用场景优化\n(学校/家庭/咨询)', '#1abc9c', 1),
    ]
    
    for name, color, y in adaptations:
        box = FancyBboxPatch((8, y), 5, 1.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(box)
        ax.text(10.5, y+0.75, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 底部说明
    ax.text(7, 0.3, '策略有效性: 基本情绪表达具有跨年龄普遍性 + 分类体系针对青少年需求设计 + 深度学习特征具有迁移能力',
           ha='center', va='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '07_迁移学习策略.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 8. 应用场景图
# ============================================================
def generate_application_scenarios():
    """应用场景图"""
    print("生成应用场景图...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 中心系统
    center = FancyBboxPatch((5, 4), 4, 2, boxstyle="round,pad=0.1",
                             facecolor='#3498db', edgecolor='black', linewidth=3)
    ax.add_patch(center)
    ax.text(7, 5.3, '青少年情绪', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(7, 4.7, '识别系统', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 应用场景
    scenarios = [
        ('学校心理健康监测', '课堂表情分析\n作文情绪分析\n心理咨询辅助', '#2ecc71', 1, 7.5),
        ('在线教育反馈', '学习状态监测\n疲劳预警\n个性化推荐', '#e74c3c', 10, 7.5),
        ('家庭沟通辅助', '情绪日志\n沟通建议\n异常提醒', '#9b59b6', 1, 1),
        ('心理咨询支持', '情绪追踪\n风险评估\n效果评估', '#f39c12', 10, 1),
    ]
    
    for name, desc, color, x, y in scenarios:
        box = FancyBboxPatch((x, y), 3, 2, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.5, y+1.5, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        ax.text(x+1.5, y+0.7, desc, ha='center', va='center', fontsize=8, color='white')
        
        # 连接线
        ax.annotate('', xy=(7, 5), xytext=(x+1.5, y+1),
                   arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5,
                                  connectionstyle="arc3,rad=0.2"))
    
    # 标题
    ax.text(7, 9.5, '青少年情绪识别系统应用场景', ha='center', va='center',
           fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, '08_应用场景.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("生成论文叙述配套图片")
    print("="*60)
    
    generate_mental_health_status()
    generate_emotion_characteristics()
    generate_russell_4class_mapping()
    generate_4class_intervention()
    generate_hmtl_architecture()
    generate_multimodal_fusion()
    generate_transfer_learning()
    generate_application_scenarios()
    
    print("\n" + "="*60)
    print(f"所有图片已保存到: {OUTPUT_DIR}")
    print("="*60)
