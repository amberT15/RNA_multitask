import rna_model
import torch
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from torch.utils.data.dataset import random_split
from sequence_models.constants import SPECIALS,PAD,START,STOP,MASK
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import wandb
wandb.login()
torch.manual_seed(0)
RNA='ACGTN'
RNA_ALPHABET = SPECIALS+RNA
seq_data = rna_model.rna_self_mask('./data/pre-train/510/rna_seq.h5',RNA,SPECIALS)
train_data,valid_data = random_split(seq_data,[int(len(seq_data)*0.9),int(len(seq_data)*0.1)+1])

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

training_args = TrainingArguments(
    output_dir="./bert",
    overwrite_output_dir=True,
    num_train_epochs=10,
    do_train=True,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2
    ,report_to="wandb"
)

log_config = {**configuration.to_dict(),**training_args.to_dict()}
wandb.init(project="rna-selftrain", 
            config = log_config)
data_collator = rna_model.BertCollater(RNA_ALPHABET,False,False,mut_alphabet=RNA)
trainer = Trainer(model = model,
                 args = training_args, 
                train_dataset=train_data, 
                eval_dataset=valid_data,
                data_collator=data_collator)

trainer.train()