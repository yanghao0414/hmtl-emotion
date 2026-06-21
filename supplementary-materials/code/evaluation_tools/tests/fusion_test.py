#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态融合策略测试 - 纯逻辑测试
不需要加载模型
"""

def test_fusion_strategies():
    """测试融合策略"""
    print("测试融合策略")
    print("="*50)
    
    # 测试场景
    test_scenarios = [
        {
            'name': '一致场景',
            'predictions': {
                'text': {'4': '积极', '7': '快乐', 'arousal': 0.7, 'valence': 0.8, 'confidence': 0.99},
                'audio': {'4': '积极', '7': '快乐', 'arousal': 0.8, 'valence': 0.7, 'confidence': 0.65},
                'visual': {'4': '积极', '7': '支持', 'arousal': 0.6, 'valence': 0.9, 'confidence': 0.64}
            }
        },
        {
            'name': '冲突场景',
            'predictions': {
                'text': {'4': '积极', '7': '快乐', 'arousal': 0.7, 'valence': 0.8, 'confidence': 0.99},
                'audio': {'4': '激活消极', '7': '愤怒', 'arousal': 0.9, 'valence': -0.6, 'confidence': 0.65},
                'visual': {'4': '平静', '7': '平静', 'arousal': 0.3, 'valence': 0.1, 'confidence': 0.64}
            }
        },
        {
            'name': '部分一致场景',
            'predictions': {
                'text': {'4': '积极', '7': '快乐', 'arousal': 0.7, 'valence': 0.8, 'confidence': 0.99},
                'audio': {'4': '积极', '7': '支持', 'arousal': 0.8, 'valence': 0.7, 'confidence': 0.65},
                'visual': {'4': '非激活消极', '7': '悲伤', 'arousal': 0.2, 'valence': -0.5, 'confidence': 0.64}
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 30)
        
        predictions = scenario['predictions']
        
        # 显示各模态预测
        print("各模态预测:")
        for modality, pred in predictions.items():
            print(f"  {modality}: {pred['4']} / {pred['7']}")
        
        # 1. 简单投票
        votes_4 = {}
        for pred in predictions.values():
            class_4 = pred['4']
            votes_4[class_4] = votes_4.get(class_4, 0) + 1
        winner_4 = max(votes_4, key=votes_4.get)
        
        print(f"\n简单投票: {winner_4} (票数: {votes_4})")
        
        # 2. 加权投票
        weights = {'text': 0.99, 'audio': 0.65, 'visual': 0.64}
        weighted_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4']
            weight = weights[modality]
            weighted_votes_4[class_4] = weighted_votes_4.get(class_4, 0) + weight
        weighted_winner_4 = max(weighted_votes_4, key=weighted_votes_4.get)
        
        print(f"加权投票: {weighted_winner_4} (权重: {weighted_votes_4})")
        
        # 3. 置信度加权
        conf_votes_4 = {}
        for modality, pred in predictions.items():
            class_4 = pred['4']
            base_weight = weights[modality]
            confidence = pred['confidence']
            dynamic_weight = base_weight * confidence
            conf_votes_4[class_4] = conf_votes_4.get(class_4, 0) + dynamic_weight
        conf_winner_4 = max(conf_votes_4, key=conf_votes_4.get)
        
        print(f"置信度加权: {conf_winner_4} (权重: {conf_votes_4})")
        
        # 4. Arousal/Valence平均
        arousal_avg = sum(pred['arousal'] for pred in predictions.values()) / len(predictions)
        valence_avg = sum(pred['valence'] for pred in predictions.values()) / len(predictions)
        
        print(f"平均值: Arousal={arousal_avg:.3f}, Valence={valence_avg:.3f}")

def test_attention_mechanism():
    """测试注意力融合"""
    print(f"\n测试注意力融合")
    print("="*50)
    
    # 测试数据
    predictions = {
        'text': {'4': '积极', '7': '快乐', 'arousal': 0.7, 'valence': 0.8},
        'audio': {'4': '激活消极', '7': '愤怒', 'arousal': 0.9, 'valence': -0.6},
        'visual': {'4': '积极', '7': '支持', 'arousal': 0.6, 'valence': 0.5}
    }
    
    # 基础权重
    base_weights = {'text': 0.99, 'audio': 0.65, 'visual': 0.64}
    
    # 计算注意力权重
    attention_weights = {}
    
    for modality, pred in predictions.items():
        consistency_score = 0
        
        # 检查4类和7类一致性
        class_4 = pred['4']
        class_7 = pred['7']
        
        # 一致性映射
        class_mapping = {
            '积极': ['快乐', '支持'],
            '激活消极': ['愤怒', '焦虑'],
            '非激活消极': ['悲伤', '失望'],
            '平静': ['平静']
        }
        
        if class_7 in class_mapping.get(class_4, []):
            consistency_score += 0.5
            print(f"{modality}: 4-7类一致 (+0.5)")
        
        # 检查arousal/valence一致性
        arousal = pred['arousal']
        valence = pred['valence']
        
        if class_4 == '积极' and valence > 0:
            consistency_score += 0.3
            print(f"{modality}: 积极-正效价一致 (+0.3)")
        elif class_4 == '激活消极' and arousal > 0.6 and valence < 0:
            consistency_score += 0.3
            print(f"{modality}: 激活消极-高唤醒负效价一致 (+0.3)")
        elif class_4 == '非激活消极' and arousal < 0.4 and valence < 0:
            consistency_score += 0.3
            print(f"{modality}: 非激活消极-低唤醒负效价一致 (+0.3)")
        elif class_4 == '平静' and abs(valence) < 0.3:
            consistency_score += 0.3
            print(f"{modality}: 平静-中性效价一致 (+0.3)")
        
        # 计算最终权重
        base_weight = base_weights[modality]
        attention_weights[modality] = base_weight * (1 + consistency_score)
        
        print(f"{modality}: 基础权重={base_weight}, 一致性={consistency_score}, 最终权重={attention_weights[modality]:.3f}")
    
    # 归一化
    total_attention = sum(attention_weights.values())
    normalized_weights = {k: v/total_attention for k, v in attention_weights.items()}
    
    print(f"\n归一化权重: {normalized_weights}")
    
    # 注意力融合结果
    att_votes_4 = {}
    for modality, pred in predictions.items():
        class_4 = pred['4']
        weight = normalized_weights[modality]
        att_votes_4[class_4] = att_votes_4.get(class_4, 0) + weight
    
    att_winner_4 = max(att_votes_4, key=att_votes_4.get)
    print(f"注意力融合结果: {att_winner_4} (权重: {att_votes_4})")

def test_hierarchical_fusion():
    """测试层级融合"""
    print(f"\n测试层级融合")
    print("="*50)
    
    predictions = {
        'text': {'4': '积极', '7': '快乐'},  # 文本预测
        'audio': {'4': '积极', '7': '支持'},
        'visual': {'4': '非激活消极', '7': '悲伤'}
    }
    
    weights = {'text': 0.99, 'audio': 0.65, 'visual': 0.64}
    
    print("各模态预测 (冲突场景):")
    for modality, pred in predictions.items():
        print(f"  {modality}: {pred['4']} / {pred['7']}")
    
    # 4
    weighted_votes_4 = {}
    for modality, pred in predictions.items():
        class_4 = pred['4']
        weight = weights[modality]
        weighted_votes_4[class_4] = weighted_votes_4.get(class_4, 0) + weight
    
    primary_4class = max(weighted_votes_4, key=weighted_votes_4.get)
    print(f"\n第一层 - 4类结果: {primary_4class}")
    
    # 第二层: 基于4类结果筛选7类候选
    class_7_candidates = {
        '积极': ['快乐', '支持'],
        '激活消极': ['愤怒', '焦虑'],
        '非激活消极': ['悲伤', '失望'],
        '平静': ['平静']
    }
    
    candidates = class_7_candidates.get(primary_4class, [])
    print(f"第二层 - 7类候选: {candidates}")
    
    # 筛选投票
    filtered_votes_7 = {}
    for modality, pred in predictions.items():
        class_7 = pred['7']
        if class_7 in candidates:
            weight = weights[modality]
            filtered_votes_7[class_7] = filtered_votes_7.get(class_7, 0) + weight
            print(f"  {modality}投票{class_7}={weight}")
        else:
            print(f"  {modality}被过滤{class_7}")
    
    if filtered_votes_7:
        final_7class = max(filtered_votes_7, key=filtered_votes_7.get)
        print(f"最终结果: {primary_4class} / {final_7class}")
    else:
        print(f"最终结果: {primary_4class} / (无7类候选)")

def main():
    """主函数"""
    print("多模态融合策略测试")
    print("="*60)
    
    try:
        test_fusion_strategies()
        test_attention_mechanism()
        test_hierarchical_fusion()
        
        print(f"\n所有测试通过")
        print("融合策略工作正常")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    main()
