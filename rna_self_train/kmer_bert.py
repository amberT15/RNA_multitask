import wandb
import rna_model
import utils
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments
from dna_tokenizer import rnabert_maskwrapper,DNATokenizer

import os
os.environ['WANDB_PROJECT'] = 'rna_MLM'
os.environ['WANDB_LOG_MODEL'] = 'true'
### RUN_DATASET_TYPE
runtype = 'context'
#runtype = '6mer'

tokenizer = DNATokenizer('./rna_self_train/vocab.txt')
if runtype == 'context':
    decay_rate=0.05
    data_dir = './data/pre-train/510/rna_seq.h5'
    train_data = rna_model.rna_context(data_dir,'train',6,tokenizer)
    valid_data = rna_model.rna_context(data_dir,'valid',6,tokenizer)
    data_collator = rnabert_maskwrapper(tokenizer,decay_rate,extend = True)

elif runtype == '6mer':
    decay_rate = 0.15
    data_dir = './data/pre-train/510_6/rna_seq.h5'
    train_data = rna_model.rna_kmer(data_dir,'train',6,tokenizer)
    valid_data = rna_model.rna_kmer(data_dir,'valid',6,tokenizer)
    data_collator = rnabert_maskwrapper(tokenizer,decay_rate,extend = False)

run_name = runtype+'_dr'+str(decay_rate)+'_bert_v0'

# Initializing a Bert configuration
configuration = BertConfig(
                            vocab_size = tokenizer.vocab_size,
                            pad_token_id = tokenizer.pad_token_id,
                            eos_token_id = tokenizer.sep_token_id,
                            bos_token_id = tokenizer.cls_token_id,
                            type_vocab_size = 1,
                            )
# Initializing a model from the configuration
model = BertForMaskedLM(configuration)

args = utils.parse_args()

if args.local_rank == 0:
    training_args = TrainingArguments(
        output_dir = './wandb/'+run_name,
        run_name = run_name,
        num_train_epochs=70,
        do_train=True,
        learning_rate = 6e-04,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        weight_decay = 0.01,
        warmup_steps = 500,
        lr_scheduler_type = 'linear',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps = 40,
        per_device_train_batch_size=32,
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

    wandb.config.update({'decay_rate':decay_rate,'dataset':data_dir})   

else:
    training_args = TrainingArguments(
        output_dir = './wandb/'+run_name,
        run_name = run_name,
        num_train_epochs=70,
        do_train=True,
        learning_rate = 6e-04,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        weight_decay = 0.01,
        warmup_steps = 500,
        lr_scheduler_type = 'linear',
        evaluation_strategy = 'steps',
        gradient_accumulation_steps = 40,
        per_device_train_batch_size=32,
        logging_steps = 80,
        eval_steps = 500,
        save_total_limit=2,
        ddp_find_unused_parameters = False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(model = model, 
                    args = training_args, 
                    train_dataset=train_data, 
                    eval_dataset=valid_data,
                    data_collator=data_collator)


trainer.train()