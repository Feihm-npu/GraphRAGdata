import pickle
import os
import numpy as np

# 设置路径
embedding_dir = "/home/feihm/llm-fei/Data/NQ/wikipedia_embeddings"

# 随便选一个 shard 文件（如 passages_00）
filename = os.path.join(embedding_dir, "passages_00")

# 加载数据
with open(filename, "rb") as f:
    ids, embeddings = pickle.load(f)

# 打印格式和部分内容
print("✅ 加载成功！")
print(f"Passage 数量: {len(ids)}")
print(f"Embedding 维度: {embeddings.shape}")  # 例如 (50000, 768)

# 示例打印前5条
for i in range(5):
    print(f"ID: {ids[i]}")
    print(f"Embedding[:5]: {embeddings[i][:5]}")  # 只打印前5个元素
    print("---")
