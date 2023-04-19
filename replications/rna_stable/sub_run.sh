#!/bin/bash
#insert_dataset','gpn_finetune_embed','gpn_human_embed','gpn_plant_embed'
# CUDA_VISIBLE_DEVICES=2 python subsample.py insert_dataset &
# CUDA_VISIBLE_DEVICES=3 python subsample.py gpn_finetune_embed &
# CUDA_VISIBLE_DEVICES=4 python subsample.py gpn_human_embed &
# CUDA_VISIBLE_DEVICES=5 python subsample.py gpn_plant_embed &
CUDA_VISIBLE_DEVICES=3 python subsample.py convformer_species16_embed &
CUDA_VISIBLE_DEVICES=2 python subsample.py convformer_species7_embed &
CUDA_VISIBLE_DEVICES=1 python subsample.py convformer_species20_embed &
wait
