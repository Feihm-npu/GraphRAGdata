export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=1,4

lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tensor_parallel_size=2,dtype=auto   \
    --tasks graphrag_qa_advanced \
    --batch_size auto:8 \
    --write_out \
    --log_samples \
    --output_path GraphRAG \
    --limit 100

## Qwen/Qwen2.5-7B-Instruct GraphRAG
# |       Tasks        |Version|     Filter      |n-shot|  Metric   |   |Value|   |Stderr|
# |--------------------|------:|-----------------|-----:|-----------|---|----:|---|-----:|
# |graphrag_qa_advanced|      1|remove_whitespace|     0|exact_match|↑  | 0.33|±  |0.0473|

## Qwen/Qwen2.5-7B-Instruct RAG