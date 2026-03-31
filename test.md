```sh
NCCL_P2P_DISABLE=1 PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=29506 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/orig

PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=29506 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/orig

# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=29506 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/orig

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=34578 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/orig

NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=34578 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/orig

NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 PYTHONPATH=src:externals/waymo-open-dataset/src:externals/TATS/tats/fvd torchrun --nproc_per_node=7 --master_port=34578 src/dwm/evaluate.py -c configs/ctsd/single_dataset/ctsd_35_tirda_bm_nusc_a.json -o output/no_hdmap
```

```py
# /media/data4/sumq/opendwm/OpenDWM/src/dwm/datasets/nuscenes.py
# NOTE HACK
```
