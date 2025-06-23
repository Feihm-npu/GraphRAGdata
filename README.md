# Generate Graph RAG dataset based on Wiki passages.


## Roadmap
```
[Step 0] 使用 Contriever 检索得到 nq.json（每个问题 → 100 passages） ✅你已完成
     ↓
[Step 1] 对每个 question 的 100 个 passage 文本进行预处理（分句、清洗）
     ↓
[Step 2] 使用 LLM 或轻量 NER 模型，批量提取实体
     ↓
[Step 3] 使用 LLM 或规则系统抽取关系（生成三元组）
     ↓
[Step 4] 构建每个 question 对应的知识图谱（GraphRAG 支持结构）
     ↓
[Step 5] 基于该 GraphRAG 图谱检索相关上下文或作为输入支持问答生成
     ↓
[Step 6] 使用 LLM 生成答案（可加 GraphRAG-augmented prompt）

```
**Details ragarding building GraphRAG**
```
问题 --> Contriever 语义检索 top-100 wiki 文档
     --> 对这 100 文档构建 Graph（实体+关系）
     --> 送入 GraphRAG 检索 context / 增强 LLM
```


## Detailed steps
1. The preprocessing has already been done in the [Contriver repository](https://github.com/facebookresearch/contriever).
The embeddings can be downloaded:
    ```sh
    wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar
    wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
    ```
    Retrieve top-100 passages:
    ```sh
    python passage_retrieval.py \
        --model_name_or_path facebook/contriever \
        --passages psgs_w100.tsv \
        --passages_embeddings "contriever_embeddings/*" \
        --data nq_dir/test.json \
        --output_dir contriever_nq \
    ```
