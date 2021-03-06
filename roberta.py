import rna_model
import torch
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import random_split
from sequence_models.constants import SPECIALS,PAD,START,STOP,MASK
import utils
import wandb

torch.manual_seed(0)
RNA='ACGTN'
RNA_ALPHABET = SPECIALS+RNA
train_data = rna_model.rna_self_mask('./data/pre-train/510/rna_seq.h5','train')
valid_data = rna_model.rna_self_mask('./data/pre-train/510/rna_seq.h5','valid')

# Initializing a RoBERTa configuration
# Initializing a RoBERTa configuration
configuration = RobertaConfig(vocab_size = len(RNA_ALPHABET),
                            pad_token_id = RNA_ALPHABET.index(PAD),
                            eos_token_id = RNA_ALPHABET.index(STOP),
                            bos_token_id = RNA_ALPHABET.index(START),
                            type_vocab_size = 1,
                            layer_norm_eps = 1e-05,
                            max_position_embeddings = 514 )
# Initializing a model from the configuration
model = RobertaForMaskedLM(configuration)
args = utils.parse_args()

if args.local_rank == 0:
    training_args = TrainingArguments(
        output_dir="./bert",
        overwrite_output_dir=True,
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        ddp_find_unused_parameters = False
        ,report_to="wandb"
    )
    log_config = {**configuration.to_dict(),**training_args.to_dict()}
    #wandb.login()
    wandb.init(project="rna-selftrain", 
                config = log_config)
else:
    training_args = TrainingArguments(
        output_dir="./bert",
        overwrite_output_dir=True,
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        ddp_find_unused_parameters = False
    )


data_collator = rna_model.BertCollater(RNA_ALPHABET,False,False,mut_alphabet=RNA)
trainer = Trainer(model = model,
                 args = training_args, 
                train_dataset=train_data, 
                eval_dataset=valid_data,
                data_collator=data_collator)

trainer.train()