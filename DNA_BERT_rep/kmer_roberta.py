import transformers
import sys
import wandb
sys.path.append('../')
sys.path.append('./')
#sys.path.append('/home/amber/multitask_RNA/replications/DNABERT/examples/')

from dnabert_datastruct import mask_tokens
from dnabert_datastruct import DNATokenizer

from torch.utils.data import DataLoader
import rna_model
import importlib
import numpy as np
import torch
import utils
from transformers import RobertaConfig, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments


class arg():
    def __init__(self,prb):
        self.mlm_probability = prb

class rnabert_maskwrapper():
    def __init__(self,tokenizer,prob_arg) -> None:
        self.tokenizer = tokenizer
        self.prb = prob_arg
    def __call__(self, batch_entry):
        batch_entry = torch.from_numpy(np.array(batch_entry))
        input,label = mask_tokens(batch_entry,self.tokenizer,arg(self.prb))
        return{'input_ids':input,'labels':label}

importlib.reload(rna_model)
tokenizer = DNATokenizer.from_pretrained('dna6')
train_data = rna_model.rna_kmer('../data/pre-train/510/rna_seq.h5','train',6,tokenizer)
valid_data = rna_model.rna_kmer('../data/pre-train/510/rna_seq.h5','valid',6,tokenizer)
data_collator = rnabert_maskwrapper(tokenizer,0.15)

# Initializing a RoBERTa configuration
configuration = RobertaConfig(vocab_size = tokenizer.vocab_size,
                            pad_token_id = tokenizer.pad_token_id,
                            eos_token_id = tokenizer.sep_token_id,
                            bos_token_id = tokenizer.cls_token_id,
                            type_vocab_size = 1,
                            layer_norm_eps = 1e-05,
                            max_position_embeddings = 514 )
# Initializing a model from the configuration
model = RobertaForMaskedLM(configuration)

args = utils.parse_args()

if args.local_rank == 0:
    training_args = TrainingArguments(
        output_dir="./6mer-roberta",
        overwrite_output_dir=True,
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        ddp_find_unused_parameters = False,
        report_to="wandb"
    )
    log_config = {**configuration.to_dict(),**training_args.to_dict()}
    run = wandb.init(entity='ambert',project="rna-selftrain",
                config = log_config)
else:
    training_args = TrainingArguments(
        output_dir="./6mer-roberta",
        overwrite_output_dir=True,
        num_train_epochs=10,
        do_train=True,
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        ddp_find_unused_parameters = False
    )

trainer = Trainer(model = model, 
                args = training_args, 
                train_dataset=train_data, 
                eval_dataset=valid_data,
                data_collator=data_collator)

trainer.train()