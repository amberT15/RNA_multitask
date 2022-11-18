import pytorch_lightning as pl
import sys
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from sequence_models.collaters import  MLMCollater
from pytorch_lightning.loggers import WandbLogger
from sequence_models.constants import SPECIALS
import carp_models
sys.path.append('/home/amber/multitask_RNA/rna_self_train/')
from dna_tokenizer import DNATokenizer,carp_maskwrapper
import rna_model

def main():
        vocab = sys.argv[1]
        if vocab == 'base':
                RNA='ACGTN'
                RNA_ALPHABET = RNA+SPECIALS
                len_vocab = len(RNA_ALPHABET)
                padding_idx = RNA_ALPHABET.index('-')
                train_data = carp_models.rna_self_mask('/home/amber/multitask_RNA/data/pre-train/context/rna_seq.h5','train')
                valid_data = carp_models.rna_self_mask('/home/amber/multitask_RNA/data/pre-train/context/rna_seq.h5','valid')
                collater = MLMCollater(RNA_ALPHABET,True,False,mut_alphabet=RNA)

        elif vocab == 'kmer':
                tokenizer = DNATokenizer('/home/amber/multitask_RNA/rna_self_train/vocab.txt')
                decay_rate = 0.15
                len_vocab = 4101
                padding_idx = tokenizer.pad_token_id
                data_dir = '/home/amber/multitask_RNA/data/pre-train/6mer/rna_seq.h5'
                train_data = rna_model.rna_kmer(data_dir,'train',6,tokenizer)
                valid_data = rna_model.rna_kmer(data_dir,'valid',6,tokenizer)
                collater = carp_maskwrapper(tokenizer,decay_rate,extend = False)

        elif vocab == 'context':
                tokenizer = DNATokenizer('/home/amber/multitask_RNA/rna_self_train/vocab.txt')
                decay_rate=0.05
                len_vocab = 4101
                padding_idx = tokenizer.pad_token_id
                data_dir = '/home/amber/multitask_RNA/data/pre-train/context/rna_seq.h5'
                train_data = rna_model.rna_context(data_dir,'train',6,tokenizer)
                valid_data = rna_model.rna_context(data_dir,'valid',6,tokenizer)
                collater = carp_maskwrapper(tokenizer,decay_rate,extend = True)
        else:
                raise ValueError('please pass in valid vocab type')
                
        train_loader = DataLoader(train_data, num_workers=4,collate_fn = collater,batch_size = 64)
        valid_loader = DataLoader(valid_data, num_workers=4,collate_fn = collater,batch_size = 64)

        #Seed for distributed training
        pl.seed_everything(42)

        #Set hyperparameters for model building
        config={'model':'ByteNetLM',
                'lr':1e-3,
                'n_tokens':len_vocab,
                'd_embedding' : 8, # dimension of embedding
                'd_model': 1024, # dimension to use within ByteNet model, //2 every layer
                'n_layers' : 32, # number of layers of ByteNet block
                'activation': 'gelu',
                'kernel_size' : 5, # the kernel width
                'r' : 128, # used to calculate dilation factor
                'padding_idx' : padding_idx ,# location of padding token in ordered alphabet
                'dropout' : 0.1 ,
                }

        model = carp_models.ByteNetLM(config['n_tokens'], config['d_embedding'], config['d_model'],
                                config['n_layers'], config['kernel_size'], config['r'], config['lr'],
                                padding_idx=config['padding_idx'], final_ln=True, dropout=config['dropout'])

        wandb_logger = WandbLogger(project="carp-rna",config=config,log_model=True)
        checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                                monitor="val_loss", 
                                                mode="min")
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        earlystop = EarlyStopping(monitor="val_loss",
                                mode="min",patience=7)
        trainer = pl.Trainer(gpus=[0,1],detect_anomaly=True,max_epochs=100,
                        strategy="ddp",
                        logger = wandb_logger,
                        callbacks=[checkpoint_callback,
                        earlystop,lr_monitor])
        #trainer.tune(model,train_dataloaders=train_loader,val_dataloaders = valid_loader)
        trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders = valid_loader)


if __name__ == "__main__":
    main()