import torch
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sequence_models.collaters import  MLMCollater
from pytorch_lightning.loggers import WandbLogger
from sequence_models.constants import SPECIALS
import rna_model

RNA='ACGTN'
RNA_ALPHABET = RNA+SPECIALS
seq_data = rna_model.rna_self_mask('./data/pre-train/rna_seq.h5',RNA,SPECIALS)
train_data,valid_data = random_split(seq_data,[int(len(seq_data)*0.9),int(len(seq_data)*0.1)+1])
collater = MLMCollater(RNA_ALPHABET,True,False,mut_alphabet=RNA)
train_loader = DataLoader(train_data,num_workers=4,collate_fn = collater,batch_size = 128)
valid_loader = DataLoader(valid_data,num_workers=4,collate_fn = collater,batch_size = 128)

#Set hyperparameters for model building
config={'model':'ByteNetLM',
        'lr':1e-3,
        'n_tokens':len(RNA_ALPHABET),
        'd_embedding' : 9, # dimension of embedding
        'd_model': 320, # dimension to use within ByteNet model, //2 every layer
        'n_layers' : 15, # number of layers of ByteNet block
        'kernel_size' : 5, # the kernel width
        'r' : 32, # used to calculate dilation factor
        'padding_idx' : RNA_ALPHABET.index('-') ,# location of padding token in ordered alphabet
        'dropout' : 0.1 ,
        }
model = rna_model.ByteNetLM(config['n_tokens'], config['d_embedding'], config['d_model'],
                         config['n_layers'], config['kernel_size'], config['r'], config['lr'],
                        padding_idx=config['padding_idx'], causal=False, dropout=config['dropout'])

wandb_logger = WandbLogger(project="rna-selftrain",config=config,log_model=True)
checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                        monitor="val_loss",
                                        mode="min",
                                        dirpath=wandb.run.dir,
                                        filename="best_model")
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
earlystop = EarlyStopping(monitor="val_loss",
                            mode="min",patience=3)
trainer = pl.Trainer(gpus=1,detect_anomaly=True,max_epochs=100,logger = wandb_logger,
                    callbacks=[checkpoint_callback,earlystop,lr_monitor])

trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders = valid_loader)