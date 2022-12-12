import wandb
import rna_model
import utils
from transformers import LongformerConfig, LongformerForMaskedLM
from transformers import Trainer, TrainingArguments
from dna_tokenizer import rnabert_maskwrapper,DNATokenizer

import os
os.environ['WANDB_PROJECT'] = 'rna_MLM'
os.environ['WANDB_LOG_MODEL'] = 'true'

decay_rate = 0.15
tokenizer = DNATokenizer('./vocab/base_vocab.txt',max_len=3072)
data_dir = '../data/pre-train/3070/rna_seq.h5'
train_data = rna_model.rna_self_mask(data_dir,'train',tokenizer,max_length=3072)
valid_data = rna_model.rna_self_mask(data_dir,'valid',tokenizer,max_length=3072)
data_collator = rnabert_maskwrapper(tokenizer,decay_rate,extend = False)

run_name = 'base_longformer_v0'
configuration = LongformerConfig(attention_window = 512,
                                vocab_size = tokenizer.vocab_size,
                                pad_token_id = tokenizer.pad_token_id,
                                eos_token_id = tokenizer.sep_token_id,
                                bos_token_id = tokenizer.cls_token_id,
                                type_vocab_size = 1,
                                layer_norm_eps = 1e-05,
                                max_position_embeddings = 3074
                                )

model = LongformerForMaskedLM(configuration)
args = utils.parse_args()
print(args.local_rank)
if args.local_rank == 0:
    training_args = TrainingArguments(
        output_dir = './wandb/'+run_name,
        run_name = run_name,
        num_train_epochs=70,
        do_train=True,
        learning_rate = 6e-04,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        adam_epsilon = 1e-06,
        weight_decay = 0.01,
        warmup_steps = 500,
        lr_scheduler_type = 'linear',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps = 40,
        per_device_train_batch_size=4,
        logging_steps = 80,
        eval_steps = 500,
        save_total_limit=2,
        ddp_find_unused_parameters = False,
        load_best_model_at_end=True,
        report_to="wandb"
    )
    log_config = {**configuration.to_dict(),**training_args.to_dict(),'decay_rate':decay_rate,'dataset':data_dir}
    wandb.init(entity='ambert',project="rna_MLM",
                config = log_config)

    trainer = Trainer(model = model, 
                args = training_args, 
                train_dataset=train_data, 
                eval_dataset=valid_data,
                data_collator=data_collator)

else:
    training_args = TrainingArguments(
        output_dir = './wandb/'+run_name,
        run_name = run_name,
        num_train_epochs=30,
        do_train=True,
        learning_rate = 6e-04,
        adam_beta1 = 0.9,
        adam_beta2 = 0.98,
        adam_epsilon = 1e-06,
        weight_decay = 0.01,
        warmup_steps = 500,
        lr_scheduler_type = 'linear',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps = 40,
        per_device_train_batch_size=4,
        logging_steps = 80,
        eval_steps = 500,
        save_total_limit=2,
        ddp_find_unused_parameters = False,
        load_best_model_at_end=True
    )

    trainer = Trainer(model = model, 
                    args = training_args, 
                    train_dataset=train_data, 
                    eval_dataset=valid_data,
                    data_collator=data_collator)


trainer.train()