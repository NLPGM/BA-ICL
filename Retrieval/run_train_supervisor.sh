



DATASETS=("ACE05")
K_SHOTS=(5 10 20 50)
LLM_TYPES=("Qwen2.5-3B-Instruct-GPTQ-Int4")

# 循环执行
for dataset in "${DATASETS[@]}"; do
  for k_shot in "${K_SHOTS[@]}"; do
    for llm_type in "${LLM_TYPES[@]}"; do
      echo "Running train_contrastive_retriever.py with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type"
      CUDA_VISIBLE_DEVICES=0 python train_contrastive_retriever.py \
          --dataset "$dataset" \
          --k_shot "$k_shot" \
          --llm_type "$llm_type"
    done
  done
done