import torch
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sequence_models.collaters import  MLMCollater
from pytorch_lightning.loggers import WandbLogger
from sequence_models.constants import SPECIALS
import rna_model

def main():
        RNA='ACGTN'
        RNA_ALPHABET = RNA+SPECIALS
        train_data = rna_model.rna_self_mask('./data/pre-train/510/rna_seq.h5','train')
        valid_data = rna_model.rna_self_mask('./data/pre-train/510/rna_seq.h5','valid')
        collater = MLMCollater(RNA_ALPHABET,True,False,mut_alphabet=RNA)
        train_loader = DataLoader(train_data, num_workers=4,collate_fn = collater,batch_size = 512)
        valid_loader = DataLoader(valid_data, num_workers=4,collate_fn = collater,batch_size = 512)

        #Seed for distributed training
        pl.seed_everything(42)

        #Set hyperparameters for model building
        config={'model':'ByteNetLM',
                'lr':1e-3,
                'n_tokens':len(RNA_ALPHABET),
                'd_embedding' : 8, # dimension of embedding
                'd_model': 128, # dimension to use within ByteNet model, //2 every layer
                'n_layers' : 16, # number of layers of ByteNet block
                'activation': 'gelu',
                'kernel_size' : 5, # the kernel width
                'r' : 128, # used to calculate dilation factor
                'padding_idx' : RNA_ALPHABET.index('-') ,# location of padding token in ordered alphabet
                'dropout' : 0.1 ,
                }

        model = rna_model.ByteNetLM(config['n_tokens'], config['d_embedding'], config['d_model'],
                                config['n_layers'], config['kernel_size'], config['r'], config['lr'],
                                padding_idx=config['padding_idx'], final_ln=True, dropout=config['dropout'])

        wandb_logger = WandbLogger(project="rna-selftrain",config=config,log_model="all")
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                                monitor="val_loss", 
                                                mode="min")
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        earlystop = EarlyStopping(monitor="val_loss",
                                mode="min",patience=7)
        trainer = pl.Trainer(gpus=[0,1,2,3],detect_anomaly=True,max_epochs=100,
                        strategy="ddp",
                        logger = wandb_logger,
                        callbacks=[checkpoint_callback,
                        earlystop,lr_monitor])
        #trainer.tune(model,train_dataloaders=train_loader,val_dataloaders = valid_loader)
        trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders = valid_loader)


if __name__ == "__main__":
    main()