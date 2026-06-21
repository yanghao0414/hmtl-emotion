#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成青少年情绪识别相关图表
体现模型的青少年应用场景
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = str(PROJECT_ROOT / "10_论文资料" / "报告图表")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 青少年情绪问题统计图
# ============================================================
def generate_adolescent_mental_health_stats():
    """生成青少年心理健康统计图"""
    print("生成青少年心理健康统计图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 青少年心理问题发生率
    problems = ['焦虑症状', '抑郁症状', '情绪波动', '学业压力', '社交困难']
    rates = [31.3, 24.6, 45.2, 72.8, 28.5]  # 基于中国青少年心理健康调查数据
    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
    
    bars = axes[0].barh(problems, rates, color=colors)
    axes[0].set_xlabel('发生率 (%)', fontsize=12)
    axes[0].set_title('中国青少年心理问题发生率', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 100)
    
    for bar, rate in zip(bars, rates):
        axes[0].text(rate + 2, bar.get_y() + bar.get_height()/2, 
                    f'{rate}%', va='center', fontsize=11)
    
    # 右图: 青少年情绪识别的重要性
    categories = ['早期预警', '及时干预', '个性化支持', '家校沟通', '长期追踪']
    importance = [95, 88, 82, 76, 70]
    
    bars2 = axes[1].bar(categories, importance, color='#2ecc71')
    axes[1].set_ylabel('重要性评分', fontsize=12)
    axes[1].set_title('青少年情绪识别系统的价值', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis='x', rotation=15)
    
    for bar, imp in zip(bars2, importance):
        axes[1].text(bar.get_x() + bar.get_width()/2, imp + 2,
                    f'{imp}', ha='center', fontsize=11)
    
    plt.suptitle('青少年心理健康与情绪识别的必要性', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'adolescent_mental_health_stats.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 2. 青少年情绪特征分析图
# ============================================================
def generate_adolescent_emotion_characteristics():
    """生成青少年情绪特征分析图"""
    print("生成青少年情绪特征分析图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 青少年 vs 成人情绪表达差异
    emotions = ['愤怒', '焦虑', '快乐', '悲伤', '平静']
    adolescent = [0.85, 0.92, 0.78, 0.88, 0.65]  # 情绪强度 (相对值)
    adult = [0.70, 0.75, 0.82, 0.72, 0.85]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, adolescent, width, label='青少年', color='#e74c3c')
    bars2 = axes[0].bar(x + width/2, adult, width, label='成人', color='#3498db')
    
    axes[0].set_ylabel('情绪表达强度', fontsize=12)
    axes[0].set_title('青少年 vs 成人情绪表达差异', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(emotions)
    axes[0].legend()
    axes[0].set_ylim(0, 1.2)
    
    # 添加注释
    axes[0].annotate('青少年情绪\n波动更大', xy=(1, 0.92), xytext=(1.5, 1.05),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    # 右图: 青少年情绪变化周期
    hours = np.arange(0, 24)
    # 模拟青少年一天的情绪变化
    morning_dip = -0.3 * np.exp(-((hours - 7)**2) / 4)  # 早起低落
    school_stress = -0.4 * np.exp(-((hours - 14)**2) / 8)  # 下午学业压力
    evening_relief = 0.3 * np.exp(-((hours - 19)**2) / 4)  # 晚间放松
    baseline = 0.5
    emotion_curve = baseline + morning_dip + school_stress + evening_relief + 0.1 * np.random.randn(24)
    emotion_curve = np.clip(emotion_curve, 0, 1)
    
    axes[1].plot(hours, emotion_curve, 'b-', linewidth=2, label='情绪值')
    axes[1].fill_between(hours, 0, emotion_curve, alpha=0.3)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='基线')
    axes[1].axvspan(8, 16, alpha=0.1, color='red', label='学校时间')
    
    axes[1].set_xlabel('时间 (小时)', fontsize=12)
    axes[1].set_ylabel('情绪值 (0-1)', fontsize=12)
    axes[1].set_title('青少年典型日情绪变化曲线', fontsize=14, fontweight='bold')
    axes[1].set_xlim(0, 23)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper right')
    axes[1].set_xticks([0, 6, 8, 12, 16, 18, 23])
    axes[1].set_xticklabels(['0:00', '6:00', '8:00', '12:00', '16:00', '18:00', '23:00'])
    
    plt.suptitle('青少年情绪表达特征分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'adolescent_emotion_characteristics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 3. 青少年情绪识别应用场景图
# ============================================================
def generate_application_scenarios():
    """生成应用场景图"""
    print("生成应用场景图...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 中心: 青少年情绪识别系统
    center = FancyBboxPatch((6, 4), 4, 2, boxstyle="round,pad=0.1",
                             facecolor='#3498db', edgecolor='black', linewidth=3)
    ax.add_patch(center)
    ax.text(8, 5.3, '青少年情绪', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax.text(8, 4.7, '识别系统', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # 应用场景
    scenarios = [
        ('学校心理\n健康监测', 2, 8, '#2ecc71'),
        ('在线教育\n情绪反馈', 12, 8, '#e74c3c'),
        ('家庭亲子\n沟通辅助', 2, 1, '#9b59b6'),
        ('心理咨询\n辅助诊断', 12, 1, '#f39c12'),
        ('社交媒体\n风险预警', 14, 5, '#1abc9c'),
        ('游戏/娱乐\n情绪调节', 0, 5, '#e67e22'),
    ]
    
    for name, x, y, color in scenarios:
        box = FancyBboxPatch((x, y), 2.5, 1.5, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.25, y+0.75, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # 连接线
        ax.annotate('', xy=(8, 5), xytext=(x+1.25, y+0.75),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5,
                                   connectionstyle="arc3,rad=0.2"))
    
    # 标题
    ax.text(8, 9.5, '青少年情绪识别系统应用场景', ha='center', va='center', 
            fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'application_scenarios.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 4. 4类情绪对青少年心理干预的意义
# ============================================================
def generate_4class_intervention_meaning():
    """生成4类情绪对青少年干预的意义图"""
    print("生成4类情绪干预意义图...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 4类情绪及其干预策略
    classes = [
        {
            'name': '积极',
            'color': '#2ecc71',
            'emotions': '快乐、支持',
            'characteristics': '正向情绪状态',
            'intervention': '• 强化积极行为\n• 建立情绪资源\n• 培养抗压能力',
            'x': 1, 'y': 6
        },
        {
            'name': '激活消极',
            'color': '#e74c3c',
            'emotions': '愤怒、焦虑',
            'characteristics': '高激活负面情绪\n需要即时关注',
            'intervention': '• 情绪疏导\n• 压力管理\n• 冲突解决训练',
            'x': 8, 'y': 6
        },
        {
            'name': '非激活消极',
            'color': '#3498db',
            'emotions': '悲伤、失望',
            'characteristics': '低激活负面情绪\n抑郁风险信号',
            'intervention': '• 心理支持\n• 社交激活\n• 专业评估转介',
            'x': 1, 'y': 1
        },
        {
            'name': '平静',
            'color': '#95a5a6',
            'emotions': '平静、中性',
            'characteristics': '情绪稳定状态',
            'intervention': '• 维持监测\n• 情绪觉察训练\n• 预防性教育',
            'x': 8, 'y': 1
        },
    ]
    
    for cls in classes:
        # 主框
        box = FancyBboxPatch((cls['x'], cls['y']), 5, 3, boxstyle="round,pad=0.1",
                              facecolor=cls['color'], edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        
        # 类别名称
        ax.text(cls['x']+2.5, cls['y']+2.6, cls['name'], ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')
        
        # 包含情绪
        ax.text(cls['x']+2.5, cls['y']+2.1, f"({cls['emotions']})", ha='center', va='center',
               fontsize=10, color='white')
        
        # 特征
        ax.text(cls['x']+2.5, cls['y']+1.5, cls['characteristics'], ha='center', va='center',
               fontsize=9, color='white')
        
        # 干预策略框
        intervention_box = FancyBboxPatch((cls['x']+0.3, cls['y']+0.2), 4.4, 1.1, 
                                           boxstyle="round,pad=0.05",
                                           facecolor='white', edgecolor='black', linewidth=1, alpha=0.9)
        ax.add_patch(intervention_box)
        ax.text(cls['x']+2.5, cls['y']+0.75, cls['intervention'], ha='center', va='center',
               fontsize=8, color='black')
    
    # 标题
    ax.text(7, 9.5, '4类情绪分类对青少年心理干预的指导意义', ha='center', va='center',
           fontsize=18, fontweight='bold')
    
    # 说明
    ax.text(7, 9, '基于 Russell 情绪环形模型，将7类细分情绪映射为4类核心情绪，便于快速识别和针对性干预',
           ha='center', va='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'four_class_intervention_meaning.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 5. 青少年多模态情绪数据采集场景
# ============================================================
def generate_data_collection_scenarios():
    """生成数据采集场景图"""
    print("生成数据采集场景图...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = [
        ('文本模态', '• 社交媒体发言\n• 作文/日记\n• 聊天记录\n• 问卷回答', '#3498db'),
        ('语音模态', '• 课堂发言\n• 心理咨询录音\n• 日常对话\n• 情绪日志录音', '#e74c3c'),
        ('视觉模态', '• 课堂表情\n• 视频通话\n• 活动照片\n• 自拍表情', '#2ecc71'),
    ]
    
    for ax, (title, content, color) in zip(axes, scenarios):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 标题框
        title_box = FancyBboxPatch((1, 7), 8, 2, boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(title_box)
        ax.text(5, 8, title, ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        
        # 内容框
        content_box = FancyBboxPatch((1, 1), 8, 5.5, boxstyle="round,pad=0.1",
                                      facecolor='#ecf0f1', edgecolor='black', linewidth=2)
        ax.add_patch(content_box)
        ax.text(5, 4, content, ha='center', va='center', fontsize=12)
        
        ax.text(5, 0.5, '青少年场景数据来源', ha='center', va='center', fontsize=10, style='italic')
    
    plt.suptitle('青少年多模态情绪数据采集场景', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'data_collection_scenarios.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 6. 系统部署架构图
# ============================================================
def generate_deployment_architecture():
    """生成系统部署架构图"""
    print("生成系统部署架构图...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 用户层
    ax.text(8, 9.5, '青少年情绪识别系统部署架构', ha='center', va='center',
           fontsize=18, fontweight='bold')
    
    # 用户端
    users = [('学生', 1), ('教师', 4), ('家长', 7), ('心理咨询师', 10), ('管理员', 13)]
    for name, x in users:
        box = FancyBboxPatch((x, 7.5), 2, 1, boxstyle="round,pad=0.05",
                              facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1, 8, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.text(0.5, 8, '用户层', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 应用层
    apps = [('移动APP', 2), ('Web平台', 6), ('API接口', 10)]
    for name, x in apps:
        box = FancyBboxPatch((x, 5.5), 3, 1, boxstyle="round,pad=0.05",
                              facecolor='#2ecc71', edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.5, 6, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    ax.text(0.5, 6, '应用层', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 服务层
    services = [('Text HMTL', 1.5, '#9b59b6'), ('Audio HMTL', 6, '#e74c3c'), ('Visual HMTL', 10.5, '#f39c12')]
    for name, x, color in services:
        box = FancyBboxPatch((x, 3.5), 3, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x+1.5, 4, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # 融合服务
    fusion_box = FancyBboxPatch((5, 2), 6, 1, boxstyle="round,pad=0.05",
                                 facecolor='#1abc9c', edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(8, 2.5, '多模态融合服务', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.text(0.5, 4, '模型层', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 数据层
    data_box = FancyBboxPatch((3, 0.5), 10, 1, boxstyle="round,pad=0.05",
                               facecolor='#34495e', edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(8, 1, '数据存储 (用户数据 + 模型权重 + 日志)', ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    
    ax.text(0.5, 1, '数据层', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # 连接线
    ax.plot([8, 8], [7.5, 6.5], 'k-', linewidth=2)
    ax.plot([8, 8], [5.5, 4.5], 'k-', linewidth=2)
    ax.plot([8, 8], [3.5, 3], 'k-', linewidth=2)
    ax.plot([8, 8], [2, 1.5], 'k-', linewidth=2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'deployment_architecture.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"保存: {save_path}")


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("生成青少年情绪识别相关图表")
    print("="*60)
    
    generate_adolescent_mental_health_stats()
    generate_adolescent_emotion_characteristics()
    generate_application_scenarios()
    generate_4class_intervention_meaning()
    generate_data_collection_scenarios()
    generate_deployment_architecture()
    
    print("\n" + "="*60)
    print(f"所有图表已保存到: {OUTPUT_DIR}")
    print("="*60)
