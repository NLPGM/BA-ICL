datasets=("ACE05_CN")
k_shots=(5)


for dataset in "${datasets[@]}"
do
   for k_shot in "${k_shots[@]}"
   do
           CUDA_VISIBLE_DEVICES=1 python main_original.py  --epoch 10 \
                       -lr 1e-3  \
                       --bert_lr 2e-5  \
                       --batch_size 16  \
                       --dataset $dataset  \
                       --seed 42 \
                       --k_shot $k_shot

   done
done