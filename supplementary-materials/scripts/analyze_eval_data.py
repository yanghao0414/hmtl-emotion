import json
from collections import Counter

eval_d = json.load(open(r'd:\bigcreate\05_数据文件\full_eval_set_hmtl_v2.json', 'r', encoding='utf-8'))
train_d = json.load(open(r'd:\bigcreate\05_数据文件\full_training_set_hmtl_v2.json', 'r', encoding='utf-8'))

label_names = {0: '积极', 1: '激活消极', 2: '非激活消极', 3: '平静'}

# 各类典型文本
for label in range(4):
    samples = [s for s in eval_d if s['label_4'] == label]
    print(f"\n=== {label_names[label]} (共{len(samples)}条) ===")
    for s in samples[:5]:
        emo = s['original_emotion']
        text = s['text'][:60]
        print(f"  [{emo}] {text}")

# 标签冲突检查
text_labels = {}
conflict = 0
for s in eval_d:
    t = s['text']
    if t in text_labels:
        if text_labels[t] != s['label_4']:
            conflict += 1
    else:
        text_labels[t] = s['label_4']
print(f"\n评估集内标签冲突: {conflict}")

# 文本长度
lengths = [len(s['text']) for s in eval_d]
print(f"文本平均长度: {sum(lengths)/len(lengths):.0f}字")

# 关键问题：检查文本是否有明显的情感关键词
# 看看BERT是不是靠关键词就能轻松分类
keywords_pos = ['谢谢', '感谢', '鼓励', '支持', '开心', '快乐', '高兴']
keywords_neg_act = ['愤怒', '生气', '焦虑', '紧张', '害怕', '恐惧', '担心']
keywords_neg_inact = ['悲伤', '难过', '失望', '沮丧', '无助']
keywords_calm = ['嗯', '好的', '明白', '了解', '知道了']

for name, kws, label in [
    ('积极关键词', keywords_pos, 0),
    ('激活消极关键词', keywords_neg_act, 1),
    ('非激活消极关键词', keywords_neg_inact, 2),
    ('平静关键词', keywords_calm, 3),
]:
    match = sum(1 for s in eval_d if any(k in s['text'] for k in kws))
    correct_match = sum(1 for s in eval_d if any(k in s['text'] for k in kws) and s['label_4'] == label)
    print(f"{name}: 匹配{match}条, 其中标签正确{correct_match}条")

# 最关键：看看不同对话之间的文本相似度
# 很多心理咨询对话的回复模式是相似的
print("\n=== 高频文本（出现>5次）===")
all_texts = [s['text'] for s in train_d + eval_d]
text_counts = Counter(all_texts)
for text, count in text_counts.most_common(20):
    print(f"  [{count}次] {text[:50]}")
