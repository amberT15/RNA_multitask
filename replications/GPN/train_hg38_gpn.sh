 WANDB_PROJECT=GPN_MLM_512 python ./run_mlm_custom.py \
    --report_to wandb \
    --run_name GPN_hg38_resume \
    --do_train \
    --do_eval \
    --train_fasta_path /home/amber/multitask_RNA/data/gpn/train/hg38.train.parquet \
    --validation_file /home/amber/multitask_RNA/data/gpn/test/hg38.test.512.256.parquet \
    --model_type ConvNet \
    --line_by_line True \
    --window_size 512 \
    --learning_rate 1e-4 \
    --save_strategy steps \
    --save_steps 20000 \
    --max_steps 800000 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 8 \
    --warmup_steps 10000 \
    --logging_steps 10000 \
    --output_dir GPN_hg38 \
    --tokenizer_name gonzalobenegas/gpn-arabidopsis \
    --per_device_train_batch_size 250 \
    --per_device_eval_batch_size 250 \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --weight_decay 0.01 \
    --optim adamw_torch \
    --adam_epsilon 1e-4 \
    --seed 50 \
    --prediction_loss_only True \
    --lr_scheduler_type constant_with_warmup \
#    --model_name_or_path ./GPN_hg38/checkpoint-200000/

#    --resume_from_checkpoint ./GPN_hg38/checkpoint-200000/ \
#    --model_name_or_path ./GPN_hg38/checkpoint-200000/ \
#    --config_overrides vocab_size=6 \
#    --ignore_data_skip \