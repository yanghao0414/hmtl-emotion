#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMTL模型评估和预测脚本
"""

import sys
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_model_code = _PROJECT_ROOT / "02_模型代码"
if str(_model_code) not in sys.path:
    sys.path.insert(0, str(_model_code))

import torch
import json
from transformers import BertTokenizer
from hmtl_model import HMTLEmotionModel
from hmtl_model_v2 import HMTLEmotionModelV2
from hmtl_utils import predict_original_emotion, LABEL_4_NAMES, LABEL_3_NAMES

# V2模型的7分类名称
EMOTION_7_NAMES = {
    0: '愤怒', 1: '焦虑', 2: '快乐', 3: '悲伤',
    4: '失望', 5: '支持', 6: '平静'
}


class HMTLPredictor:
    """HMTL预测器"""
    
    def __init__(self,
                 model_path: str,
                 bert_model_name: str = 'bert-base-chinese',
                 device: str = None):
        
        # 设备
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 加载模型 - 自动检测V1或V2
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检测是否为V2模型（有classifier_7）
        if any('classifier_7' in key for key in checkpoint['model_state_dict'].keys()):
            self.model = HMTLEmotionModelV2(bert_model_name).to(self.device)
            self.is_v2 = True
            print(f"✓ 检测到V2模型 (99.37%准确率)")
        else:
            self.model = HMTLEmotionModel(bert_model_name).to(self.device)
            self.is_v2 = False
            print(f"✓ 检测到V1模型 (79%准确率)")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ 模型已加载: {model_path}")
        print(f"✓ 使用设备: {self.device}")
    
    def predict(self, text: str, return_details: bool = True):
        """
        预测单条文本
        
        Args:
            text: 输入文本
            return_details: 是否返回详细信息
        
        Returns:
            dict: 预测结果
        """
        # 编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # 获取结果
        label_4_probs = torch.softmax(outputs['label_4_logits'], dim=1)[0]
        label_3_probs = torch.softmax(outputs['label_3_logits'], dim=1)[0]
        
        label_4 = torch.argmax(label_4_probs).item()
        label_3 = torch.argmax(label_3_probs).item()
        arousal = outputs['arousal'][0].item()
        valence = outputs['valence'][0].item()
        
        # V2模型直接输出7分类，V1需要推断
        if self.is_v2 and 'label_7_logits' in outputs:
            label_7_probs = torch.softmax(outputs['label_7_logits'], dim=1)[0]
            label_7 = torch.argmax(label_7_probs).item()
            emotion_7 = EMOTION_7_NAMES[label_7]
        else:
            emotion_7 = predict_original_emotion(label_4, arousal, valence)
        
        result = {
            'text': text,
            'emotion_7': emotion_7,  # 7类情绪
            'emotion_4': LABEL_4_NAMES[label_4],  # 4分类
            'polarity_3': LABEL_3_NAMES[label_3],  # 3分类极性
            'arousal': round(arousal, 3),  # 唤醒度
            'valence': round(valence, 3),  # 效价
        }
        
        if return_details:
            result['label_4_probs'] = {
                LABEL_4_NAMES[i]: round(prob.item(), 3)
                for i, prob in enumerate(label_4_probs)
            }
            result['label_3_probs'] = {
                LABEL_3_NAMES[i]: round(prob.item(), 3)
                for i, prob in enumerate(label_3_probs)
            }
            if self.is_v2 and 'label_7_logits' in outputs:
                result['label_7_probs'] = {
                    EMOTION_7_NAMES[i]: round(prob.item(), 3)
                    for i, prob in enumerate(label_7_probs)
                }
        
        return result
    
    def predict_batch(self, texts: list):
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text, return_details=False)
            results.append(result)
        return results


def evaluate_on_dataset(predictor, eval_data_path):
    """在评估集上测试"""
    print("\n" + "="*60)
    print("在评估集上测试模型")
    print("="*60)
    
    # 加载评估数据
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    correct_4 = 0
    correct_3 = 0
    correct_7 = 0
    total = len(eval_data)
    
    from collections import Counter
    confusion_7 = []  # 记录7类混淆
    
    print(f"评估样本数: {total}\n")
    
    for item in eval_data:
        text = item['text']
        true_emotion = item['original_emotion']
        true_label_4 = item['label_4']
        true_label_3 = item['label_3']
        
        # 预测
        pred = predictor.predict(text, return_details=False)
        
        # 4分类准确率
        pred_label_4 = list(LABEL_4_NAMES.keys())[
            list(LABEL_4_NAMES.values()).index(pred['emotion_4'])
        ]
        if pred_label_4 == true_label_4:
            correct_4 += 1
        
        # 3分类准确率
        pred_label_3 = list(LABEL_3_NAMES.keys())[
            list(LABEL_3_NAMES.values()).index(pred['polarity_3'])
        ]
        if pred_label_3 == true_label_3:
            correct_3 += 1
        
        # 7类准确率
        if pred['emotion_7'] == true_emotion:
            correct_7 += 1
        else:
            confusion_7.append(f"{true_emotion} → {pred['emotion_7']}")
    
    # 打印结果
    print(f"4分类准确率: {correct_4}/{total} = {correct_4/total:.2%}")
    print(f"3分类准确率: {correct_3}/{total} = {correct_3/total:.2%}")
    print(f"7类准确率: {correct_7}/{total} = {correct_7/total:.2%}")
    
    # 打印混淆
    if confusion_7:
        print(f"\nTop 10 混淆情况:")
        confusion_counter = Counter(confusion_7)
        for pair, count in confusion_counter.most_common(10):
            print(f"  {pair}: {count}次")
    
    return {
        'acc_4': correct_4/total,
        'acc_3': correct_3/total,
        'acc_7': correct_7/total
    }


def demo_predictions():
    """演示预测"""
    print("\n" + "="*60)
    print("HMTL模型预测演示")
    print("="*60)
    
    model_path = str(_PROJECT_ROOT / "06_模型文件" / "hmtl_models" / "best_model.pt")
    predictor = HMTLPredictor(model_path)
    
    test_texts = [
        "我很开心",
        "我很生气",
        "我很担心",
        "我很难过",
        "太失望了",
        "谢谢你的帮助",
        "心里很平静"
    ]
    
    print("\n预测结果:\n")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"文本: {result['text']}")
        print(f"  7类情绪: {result['emotion_7']}")
        print(f"  4分类: {result['emotion_4']}")
        print(f"  3分类极性: {result['polarity_3']}")
        print(f"  唤醒度: {result['arousal']:.3f}")
        print(f"  效价: {result['valence']:.3f}")
        print()


def main():
    """主函数"""
    import os
    
    model_path = str(_PROJECT_ROOT / "06_模型文件" / "hmtl_models" / "best_model.pt")
    eval_path = str(_PROJECT_ROOT / "05_数据文件" / "eval_set_hmtl.json")
    
    if not os.path.exists(model_path):
        print("✗ 模型不存在，请先训练模型!")
        print("  运行: python hmtl_train.py")
        return
    
    # 创建预测器
    predictor = HMTLPredictor(model_path)
    
    # 在评估集上测试
    if os.path.exists(eval_path):
        evaluate_on_dataset(predictor, eval_path)
    
    # 演示预测
    demo_predictions()


if __name__ == "__main__":
    main()
