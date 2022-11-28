 WANDB_PROJECT=GPN_MLM_512 python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name GPN_human_2M \
    --do_train \
    --do_eval \
    --train_fasta_path /grid/koo/home/ztang/multitask_RNA/data/gpn/train/hg38.train.parquet \
    --validation_file /grid/koo/home/ztang/multitask_RNA/data/gpn/test/hg38.test.512.256.parquet \
    --model_type ConvNet \
    --config_overrides vocab_size=6 \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-3 \
    --save_strategy steps \
    --save_steps 100000 \
    --max_steps 2000000 \
    --evaluation_strategy steps \
    --eval_steps 50000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 50000 \
    --output_dir GPN_human \
    --tokenizer_name gonzalobenegas/gpn-arabidopsis \
    --per_device_train_batch_size 250 \
    --per_device_eval_batch_size 250 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 49 \
    --prediction_loss_only True \
    --lr_scheduler_type constant_with_warmup \
#    --ignore_data_skip \
#    --model_name_or_path ./GPN_hg38/checkpoint-200000/

#    --resume_from_checkpoint ./GPN_hg38/checkpoint-200000/ \
#    --model_name_or_path ./GPN_hg38/checkpoint-200000/ \
#    --config_overrides vocab_size=6 \
#    --ignore_data_skip \