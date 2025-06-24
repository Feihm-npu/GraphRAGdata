export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,4

lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,tensor_parallel_size=2,dtype=auto   \
    --tasks graphrag_qa_advanced \
    --batch_size auto:8