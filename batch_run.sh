


##############################################################################################333
##############################################################################################333
##############################################################################################333

DATASETS=("ACE05")
K_SHOTS=(5)
LLM_TYPES=("Qwen2.5-7B-Instruct-GPTQ-Int4")
RETRIEVAL_STAGE1_MODES=('task_specific_retrieval' 'simcse_retrieval' 'random')
TOPK=5
export LLM_TEMPERATURE=0.1
RUN_IDS=(0 1 2)


# ── 计时相关 ──────────────────────────────────────────────
SCRIPT_START=$(date +%s)
TOTAL_ITERS=$(( ${#RETRIEVAL_STAGE1_MODES[@]} * ${#DATASETS[@]} \
             * ${#K_SHOTS[@]} * ${#LLM_TYPES[@]} * ${#RUN_IDS[@]} ))
COMPLETED=0

format_duration() {
 local secs=$1
 local h=$(( secs / 3600 ))
 local m=$(( (secs % 3600) / 60 ))
 local s=$(( secs % 60 ))
 printf "%02dh %02dm %02ds" $h $m $s
}
# ──────────────────────────────────────────────────────────

#
#
#
for dataset in "${DATASETS[@]}"; do
 for k_shot in "${K_SHOTS[@]}"; do
   for llm_type in "${LLM_TYPES[@]}"; do
     echo "Running prompt_explanations.py with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type"
     CUDA_VISIBLE_DEVICES=0 python prompt_explanations.py --dataset "$dataset" --k_shot "$k_shot" --llm_type "$llm_type"

     #下面的 do_intervention_ablation.py 把非ablation和ablation的结果同时完成
     echo "Running do_intervention_ablation.py with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type"
     CUDA_VISIBLE_DEVICES=0 python do_intervention_ablation.py --dataset "$dataset" --k_shot "$k_shot" --llm_type "$llm_type"
   done
 done
done

#
######################################################
# 训练监督器

cd Retrieval


# 训练task-specific baseline 所需要的检索器 （每个数据集训练一次即可）
for dataset in "${DATASETS[@]}"
do
  for k_shot in "${K_SHOTS[@]}"
  do
          CUDA_VISIBLE_DEVICES=0 python main_original.py  --epoch 10 \
                      -lr 1e-3  \
                      --bert_lr 2e-5  \
                      --batch_size 16  \
                      --dataset $dataset  \
                      --seed 42 \
                      --k_shot $k_shot

  done
done


for dataset in "${DATASETS[@]}"; do
 for k_shot in "${K_SHOTS[@]}"; do
   for llm_type in "${LLM_TYPES[@]}"; do
     echo "Running train_contrastive_retriever.py with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type"
     CUDA_VISIBLE_DEVICES=0 python train_contrastive_retriever.py \
         --dataset "$dataset" \
         --k_shot "$k_shot" \
         --llm_type "$llm_type" \
         --train_epochs 5 --train_batch_size 16
   done
 done
done

cd ..


######################################################
# 推理阶段检测与反馈


for dataset in "${DATASETS[@]}"; do
 for retrieval_mode in "${RETRIEVAL_STAGE1_MODES[@]}"; do
   for k_shot in "${K_SHOTS[@]}"; do
     for llm_type in "${LLM_TYPES[@]}"; do
       for run_id in "${RUN_IDS[@]}"; do

         echo "stage1-正常执行"
         echo "Running main.py (stage1) with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type, mode=$retrieval_mode, topk=$TOPK"
         CUDA_VISIBLE_DEVICES=0,1 python main.py --run_id "$run_id"\
             --dataset "$dataset" \
             --k_shot "$k_shot" \
             --llm_type "$llm_type" \
             --retrieval_stage1_mode "$retrieval_mode" \
             --demo_num "$TOPK" \
             --biasx_retriever_topk "$TOPK" \
             --do_stage1


         echo "stage2-正常执行Ours方法"
         echo "Running 【main.py (stage2)我们的方法】 with dataset=$dataset, k_shot=$k_shot, llm_type=$llm_type, mode=$retrieval_mode, topk=$TOPK"
         CUDA_VISIBLE_DEVICES=0,1 python main.py --run_id "$run_id"\
             --dataset "$dataset" \
             --k_shot "$k_shot" \
             --llm_type "$llm_type" \
             --retrieval_stage1_mode "$retrieval_mode" \
             --demo_num "$TOPK" \
             --biasx_retriever_topk "$TOPK" \
             --do_stage2

         # ── 循环末：进度统计 ────────────────────────────────
         COMPLETED=$(( COMPLETED + 1 ))
         NOW=$(date +%s)
         ELAPSED=$(( NOW - SCRIPT_START ))
         AVG_PER_ITER=$(( ELAPSED / COMPLETED ))
         REMAINING=$(( AVG_PER_ITER * (TOTAL_ITERS - COMPLETED) ))

         echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         echo "  ✅  iter ${COMPLETED}/${TOTAL_ITERS}  |  mode=${retrieval_mode}  dataset=${dataset}  k=${k_shot}  llm=${llm_type}  run=${run_id}"
         echo "  🕐  当前时间：$(date '+%Y-%m-%d %H:%M:%S')"
         echo "  ⏱   已消耗：$(format_duration $ELAPSED)"
         echo "  🔮  预估剩余：$(format_duration $REMAINING)  (avg $(format_duration $AVG_PER_ITER)/iter)"
         echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
         # ──────────────────────────────────────────────────────


       done
     done
   done
 done
done
