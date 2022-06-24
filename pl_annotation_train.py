
import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import h5py
import sys
import rna_model
import numpy as np

def train_model(config):
    #Task label order:[donor,acceptor,padding,utr,cds,exon]
    L = config['L']
    train_loader = DataLoader(rna_model.h5dataset(config['data'],'train')
                    ,num_workers=4,pin_memory=True,batch_size = config['batch_size'])
    valid_loader = DataLoader(rna_model.h5dataset(config['data'],'valid')
                    ,num_workers=4,pin_memory=True,batch_size = config['batch_size'])

    model = eval('rna_model.'+config['model'])(config['input_shape'],config['output_shape'],L,config['lr'])
    wandb_logger = WandbLogger(config=config,log_model=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                        monitor="val_loss",
                                        mode="min",
                                        dirpath=wandb.run.dir,
                                        filename="best_model")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    earlystop = EarlyStopping(monitor="val_loss",
                            mode="min",patience=config['patience'])
    trainer = pl.Trainer(
                        gpus=1,detect_anomaly=True,max_epochs=config['epochs'],logger = wandb_logger,
                        callbacks=[checkpoint_callback,earlystop,lr_monitor]
                        )
    trainer.fit(model=model, train_dataloaders=train_loader,val_dataloaders = valid_loader )


def train_config(config=None):
    with wandb.init(config=config):
        config = wandb.config
        history = train_model(config=config)

def main():
    exp_id = sys.argv[1]
    exp_n = sys.argv[2]
    if 'sweeps' in exp_id:
        exp_id = '/'.join(exp_id.split('/sweeps/'))
    else:
        raise BaseException('Sweep ID invalid!')
    sweep_id = exp_id
    wandb.login()
    wandb.agent(sweep_id, train_config, count=exp_n)

# __main__
################################################################################
if __name__ == '__main__':
    main()
