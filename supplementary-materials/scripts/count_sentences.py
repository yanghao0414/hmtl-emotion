"""统计原始对话数据中的句子级标注数量和标签分布"""
import json, os
from collections import Counter

path = r'd:\silent like onion\03_数据\文本已标注数据'
files = os.listdir(path)

total_sents = 0
label_counter = Counter()
file_count = 0

for f in files:
    try:
        d = json.load(open(os.path.join(path, f), 'r', encoding='utf-8'))
        clients = [m for m in d if m.get('role') == 'client' and 'annotation' in m]
        for m in clients:
            for a in m['annotation']:
                total_sents += 1
                label_counter[a['label']] += 1
        file_count += 1
    except Exception as e:
        print(f"跳过 {f}: {e}")

print(f"处理文件数: {file_count}")
print(f"句子级标注总数: {total_sents}")
print(f"\n标签分布:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count} ({count/total_sents*100:.1f}%)")
