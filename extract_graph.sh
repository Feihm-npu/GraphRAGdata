set -e

export CUDA_VISIBLE_DEVICES=0,1,4,5
WORLD_SIZE=4
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



EXTRACT_MODEL=Qwen/Qwen2.5-7B-Instruct

DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/
DS=train

TARGET_DIR_PATH=graph_data

python extract.py \
    --data_file ${DATASET_PATH}${DS}.json \
    --model_name ${EXTRACT_MODEL} \
    --world_size ${WORLD_SIZE} \
    --dest_dir ${TARGET_DIR_PATH} \
    --num_proc 32 \
    --batch_size 8 

