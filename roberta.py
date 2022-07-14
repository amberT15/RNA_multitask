import rna_model
import torch
import pytorch_lightning as pl
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from sequence_models.constants import SPECIALS
import wandb
wandb.login()

RNA='ACGTN'
RNA_ALPHABET = RNA+SPECIALS
seq_data = rna_model.bert_data('./data/pre-train/rna_seq.h5',RNA,SPECIALS)
train_data,valid_data = random_split(seq_data,[int(len(seq_data)*0.9),int(len(seq_data)*0.1)+1])

# Initializing a RoBERTa configuration

configuration = RobertaConfig(vocab_size = len(RNA_ALPHABET))
configuration.vocab_size = len(RNA_ALPHABET)
# Initializing a model from the configuration
model = RobertaForMaskedLM(configuration)

training_args = TrainingArguments(
    output_dir="./bert",
    overwrite_output_dir=True,
    num_train_epochs=10,
    do_train=True,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    report_to="wandb"
)

log_config = {**configuration.to_dict(),**training_args.to_dict()}
wandb.init(project="rna-selftrain", 
            config = log_config)

trainer = Trainer(model = model, args = training_args, train_dataset=train_data, eval_dataset=valid_data)

trainer.train()