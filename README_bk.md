# Generate Graph RAG dataset based on Wiki passages.

## Roadmap
```
[Step 1] 文本预处理（并行分句 + 清洗）
     ↓
[Step 2] 批量生成 embedding（用于 VDB）
     ↓
[Step 3] 批量实体识别（NER，GPU批推）
     ↓
[Step 4] 批量关系抽取（RE，可选用 LLM）
     ↓
[Step 5] 构建三元组（subject, relation, object）
     ↓
[Step 6] 构建知识图谱（DGL, NetworkX, LightRAG）
     ↓
[Step 7] 存储 + 检索 + LLM问答（可接 LightRAG）
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
