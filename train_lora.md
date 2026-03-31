```sh
# 二次方根 22cn2  # lr may be too large!!!
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,6 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial.json -o output/train_lora_trial

# 线性 22cn5 bf16
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/train_lora_trial_bf16

# 线性 22cn2 bf16
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,6 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/train_lora_trial_bf16

# 线性 22cn5 bf16 | no map
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,7 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/train_lora_trial_bf16

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/train_lora_trial_bf16_debug

# 线性 22cn5 bf16 | no map
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/lora_bf16_bs4_default_weight

# 线性 22cn5 bf16 减少了滑动窗口的重叠！ | no map
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc-per-node 4 --master-port 32984 src/dwm/train.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a_lora_trial_bf16.json -o output/lora_bf16_bs4_default_weight_19_10
```
