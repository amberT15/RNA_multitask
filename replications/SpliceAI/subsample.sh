#!/bin/bash

# echo "run 1"
# SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv spliceai_latest.sif python roberta_spliceai_downsample.py 0.8 ./Models/roberta_80.h5 > ./Outputs/RobertaAI_80.txt
echo "run 2"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv spliceai_latest.sif python roberta_spliceai_downsample.py 0.6 ./Models/roberta_60.h5 > ./Outputs/RobertaAI_60.txt
echo "run 3"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv spliceai_latest.sif python roberta_spliceai_downsample.py 0.4 ./Models/roberta_40.h5 > ./Outputs/RobertaAI_40.txt
echo "run 4"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv spliceai_latest.sif python roberta_spliceai_downsample.py 0.2 ./Models/roberta_20.h5 > ./Outputs/RobertaAI_20.txt


##Original Splice AI
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2,3 singularity exec --nv spliceai_latest.sif python train_model.py 400 1 ./Models/splice_100.h5 > ./Outputs/spliceai_100.txt