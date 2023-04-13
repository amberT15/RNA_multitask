import rna_model
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import wandb


config_dict = {
    'model':'conv_former',
    'hidden_size': 512,
    'attention_window': [256,256,256,256,256,256],
    'num_attention_heads': 8,
    'intermediate_size': 2048,
    'learning_rate':0.0005,
    'attention_dilation': [1,1,1,1,1,1],
    'data_dir' : '/grid/koo/home/ztang/multitask_RNA/data/species/combined_vds.h5',
    'batch_size':15,
    'gradient_accum':120,
    'warm_up_step':3000,
    'model_save_dir':'/grid/koo/home/ztang/multitask_RNA/rna_self_train/model_ckpt/large_lr_batch/'}

train_data = rna_model.longformer_dataset(config_dict['data_dir'],'train')
#valid_data = rna_model.longformer_dataset(config_dict['data_dir'],'valid')
train_loader = DataLoader(train_data,num_workers=4,batch_size = config_dict['batch_size'],shuffle=True)
#valid_loader = DataLoader(valid_data,num_workers=4,batch_size = config_dict['batch_size'])


config = rna_model.conv_former_config(**config_dict)
model = rna_model.conv_former(config)


wandb_logger = pl.loggers.WandbLogger(project="base_res_MLM",log_model=True,config = config_dict)
#wandb_logger.experiment.config.update(config_dict)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config_dict['model_save_dir'],
                                        every_n_train_steps=800)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(devices=4,accelerator='gpu',strategy='ddp',
                     logger=wandb_logger,accumulate_grad_batches=config_dict['gradient_accum'],
                     max_epochs = 100, min_steps = 3000,
                     detect_anomaly = True, log_every_n_steps=50,
                     callbacks = [checkpoint_callback,lr_monitor]
                    )

trainer.fit(model = model,train_dataloaders = train_loader)
