from datasets import load_dataset, Dataset
import json

# 加载原始数据和图数据
origin_ds = load_dataset('json', data_files='/home/feihm/llm-fei/Data/NQ/contriever_nq_all_train/train.json', split="train").select(range(2000))
graph_ds = load_dataset('json', data_files='/home/feihm/llm-fei/GraphRAGdata/graph_data/generated_outputs.jsonl', split="train",streaming=False)

print(f'len(origin_ds):{len(origin_ds)}')
print(f'len(graph_ds):{len(graph_ds)}')

assert len(origin_ds) == len(graph_ds), "mismatched size"

# 合并函数
def combined(example1, example2):
    example1 = dict(example1)  # 复制为普通 dict
    example1["triples"] = example2.get("triples", None)
    example1["passages"] = example2.get("passages", None)
    return example1

# 合并两个 dataset
combined_data = [combined(e1, e2) for e1, e2 in zip(origin_ds, graph_ds)]

# 保存为 json 文件
with open("combined_output.json", "w", encoding="utf-8") as f:
    for item in combined_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
