from posixpath import split
import torch
from torch import nn
from torch.utils.data import Dataset

import pytorch_lightning as pl
import numpy as np
import h5py

from sequence_models.convolutional import ByteNet
from sequence_models.layers import PositionFeedForward
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import  SimpleCollater,_pad
from sequence_models.constants import PAD,ALL_AAS,MASK,START,STOP

class rna_self_mask(Dataset):
    def __init__(self,h5_path,dataset):
        self.h5_file = h5py.File(h5_path, "r")[dataset][()]

    def __len__(self):
        return len(self.h5_file)

    def __getitem__(self,index):
        seq = self.h5_file[index]
        list_seq = [seq.decode("utf-8")]
        return list_seq

class ByteNetLM(pl.LightningModule):


    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r, lr,rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.0, final_ln=False, slim=True, activation='relu',
                 tie_weights=False, down_embed=True):
        super().__init__()
        self.embedder = ByteNet(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                                padding_idx=padding_idx, causal=causal, dropout=dropout, down_embed=down_embed,
                                slim=slim, activation=activation, rank=rank, n_frozen_embs=n_frozen_embs)
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

        self.loss_func = MaskedCrossEntropyLoss()
        self.lr = lr

    def forward(self, x, input_mask=None):
        e = self.embedder(x, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        loss =self.loss_func(y_hat, y, mask)
        self.log("train_loss", loss,on_step = False, on_epoch = True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        loss =self.loss_func(y_hat, y, mask)
        self.log("val_loss", loss,on_step = False, on_epoch = True, sync_dist=True)
        
        return loss
    def configure_optimizers(self):
        self.opt=torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                     mode = 'min',
                                                                     factor = 0.2,
                                                                    patience = 5,
                                                                    min_lr = 1e-7,
                                                                    verbose = True)
        schedulers =  {'optimizer':self.opt,'lr_scheduler':self.reduce_lr,'monitor':"val_loss",}
        return schedulers
