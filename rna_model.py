from posixpath import split
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import h5py
import random
import torchmetrics
import os
from transformers import PreTrainedTokenizer
from sequence_models.convolutional import ByteNet
from sequence_models.layers import PositionFeedForward
from sequence_models.losses import MaskedCrossEntropyLoss
from sequence_models.collaters import  SimpleCollater,_pad
from sequence_models.constants import PAD,ALL_AAS,MASK,START,STOP


class BertCollater(SimpleCollater):
    """Collater for masked language sequence models.

    Parameters:
        alphabet (str)
        pad (Boolean)

    Input (list): a batch of sequences as strings
    Output:
        src (torch.LongTensor): corrupted input + padding
        tgt (torch.LongTensor): input + padding
        mask (torch.LongTensor): 1 where loss should be calculated for tgt
    """

    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet

    def _prep(self, sequences):
        tgt = [START + s + STOP for s in sequences]
        sequences = [START + s + STOP for s in sequences]
        #tgt = list(sequences[:])
        src = []
        mask = []
        for seq in sequences:
            if len(seq) == 2:
                tgt.remove(seq)
                continue
            mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
            if len(mod_idx) == 0:
                mod_idx = [np.random.choice(len(seq))]  # make sure at least one aa is chosen
            seq_mod = list(seq)
            for idx in mod_idx:
                p = np.random.uniform()
                if p <= 0.10:  # do nothing
                    mod = seq[idx]
                elif 0.10 < p <= 0.20:  # replace with random amino acid
                    mod = np.random.choice([i for i in self.mut_alphabet if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
        src = [ torch.LongTensor(self.tokenizer.tokenize(s)) for s in src ]
        tgt = [ torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt ]
        pad_idx = self.tokenizer.alphabet.index(PAD)
        src = _pad(src, pad_idx)
        tgt = _pad(tgt, pad_idx)
        mask = _pad(mask, 0)
        
        label_mask = (mask==0)
        tgt[label_mask] = -100
        #return src, tgt, mask
        #return{'input_ids':src,'labels':tgt,'attention_mask':mask}
        return{'input_ids':src,'labels':tgt}

class mt_splice_data(Dataset):
    def __init__(self,h5_path,dataset):
        self.set = dataset
        self.h5_file = h5py.File(h5_path, "r")

    def __len__(self):
        return len(self.h5_file['x_'+self.set])

    def __getitem__(self,index):
        input_key = 'x_'+self.set
        fold_key = 'fold_'+self.set
        target_key = 'y_'+self.set
        inputs = self.h5_file[input_key][index].T.astype(np.float32)
        folds = self.h5_file[fold_key][index].astype(np.float32)
        targets = self.h5_file[target_key][index].T.astype(np.float32)
        return ((inputs,folds), targets)

class h5dataset(Dataset):
    def __init__(self,h5_path,dataset):
        self.set = dataset
        self.h5_file = h5py.File(h5_path, "r")

    def __len__(self):
        return len(self.h5_file['x_'+self.set])

    def __getitem__(self,index):
        input_key = 'x_'+self.set
        target_key = 'y_'+self.set
        inputs = self.h5_file[input_key][index].T.astype(np.float32)
        targets = self.h5_file[target_key][index].T.astype(np.float32)
        return (inputs, targets)

class rna_self_mask(Dataset):
    def __init__(self,h5_path,dataset):
        self.h5_file = h5py.File(h5_path, "r")[dataset][()]

    def __len__(self):
        return len(self.h5_file)

    def __getitem__(self,index):
        seq = self.h5_file[index]
        list_seq = [seq.decode("utf-8")]
        return list_seq

class rna_kmer(Dataset):
    def __init__(self,h5_path,dataset,kmer,tokenizer,max_length=512):
        self.h5_file = h5py.File(h5_path, "r")[dataset]
        self.kmer = kmer
        self.tokenizer = tokenizer
        self.maxl = max_length

    def __len__(self):
        return len(self.h5_file)

    def __getitem__(self,index):
        seq = self.h5_file[index]
        list_seq = seq.decode("utf-8")
        split_seq = ' '.join(list_seq[i:i+self.kmer] for i in range(0, len(list_seq)-self.kmer+1, 1))
        token_seq = self.tokenizer.encode(split_seq, 
                                        add_special_tokens=True, 
                                        max_length=self.maxl)
        return np.squeeze(token_seq)

class rna_long_kmer(Dataset):
    def __init__(self,h5_path,dataset,kmer,tokenizer,max_length=512):
        self.h5_file = h5py.File(h5_path, "r")[dataset]
        self.kmer = kmer
        self.tokenizer = tokenizer
        self.maxl = max_length

    def __len__(self):
        return len(self.h5_file)

    def __getitem__(self,index):
        seq = self.h5_file[index]
        list_seq = seq.decode("utf-8")
        split_seq = ' '.join(list_seq[i*self.kmer :(i+1)*self.kmer] for i in range(0, int(len(list_seq)/self.kmer)))
        token_seq = self.tokenizer.encode(split_seq, 
                                        add_special_tokens=True, 
                                        max_length=self.maxl)
        return np.squeeze(token_seq)

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

class binary_models(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #metrics
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =self.loss_func(y_hat, y)
        self.train_acc.update(y_hat, y.long())

        self.log('train_acc', self.train_acc,prog_bar=False,on_step = False, on_epoch = True)
        self.log("train_loss", loss,on_step = False, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        self.valid_acc.update(y_hat, y.long())

        self.log('valid_acc', self.valid_acc,prog_bar=False,on_step = False, on_epoch = True)
        self.log("val_loss", loss,on_step = False, on_epoch = True)
        return loss

    def configure_optimizers(self):
        self.opt=torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                     mode = 'min',
                                                                     factor = 0.2,
                                                                    patience = 3,
                                                                    min_lr = 1e-7,
                                                                    verbose = True)
        schedulers =  {'scheduler':self.reduce_lr,'monitor':"val_loss",}
        return [self.opt],schedulers

class rbp_cnn(binary_models):
    def __init__(self,exp_num,lr):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Sequential(*[
            nn.Conv1d(4,256,19,padding = 'same'),
            nn.BatchNorm1d(256)
        ])
        self.convblock = nn.Sequential(
            *[nn.MaxPool1d(8),
             nn.Dropout(0.2),
             nn.Conv1d(256,256,7,padding = 'same'),
             nn.BatchNorm1d(256),
             nn.ReLU(),
             nn.MaxPool1d(4),
             nn.Dropout(0.2),
             nn.Conv1d(256,312,7,padding = 'same'),
             nn.BatchNorm1d(312),
             nn.ReLU(),
             nn.MaxPool1d(4),
             nn.Dropout(0.2),
             nn.Conv1d(312,512,7,padding = 'same'),
             nn.BatchNorm1d(512),
             nn.ReLU(),
             nn.MaxPool1d(4),
             nn.Dropout(0.2),
             nn.Flatten(),
             nn.Linear(1536,256),
             nn.BatchNorm1d(256),
             nn.ReLU(),
             nn.Dropout(0.3)]
        )
        self.dense=nn.Linear(256,256)
        self.dense2 = nn.Linear(256,exp_num)
        self.act = nn.Sigmoid()

        self.loss_func = nn.BCELoss()

    def forward(self,x):
        nn = self.conv1(x)
        nn = torch.exp(nn)
        nn = self.convblock(nn)
        nn = self.dense(nn)
        nn=self.dense2(nn)
        output = self.act(nn)
        return output

class rbp_residual(binary_models):
    def __init__(self,exp_num,lr):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Sequential(*[
            nn.Conv1d(4,256,19,padding = 'same'),
            nn.BatchNorm1d(256)
        ])
        self.resblock = nn.Sequential(*[
            nn.MaxPool1d(8),
            dilated_residual(256,7,[1],0.1),
            nn.MaxPool1d(4),
            dilated_residual(256,7,[2],0.1),
            nn.MaxPool1d(4),
            dilated_residual(256,7,[4],0.1),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear(768,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        ])
        self.dense=nn.Linear(256,256)
        self.dense2 = nn.Linear(256,exp_num)
        self.act = nn.Sigmoid()

        self.loss_func = nn.BCELoss()

    def forward(self,x):
        nn=self.conv1(x)
        nn = torch.exp(nn)
        nn = self.resblock(nn)
        nn = self.dense(nn)
        nn=self.dense2(nn)
        output = self.act(nn)
        return output

class Splice_AI_2K(binary_models):
    def __init__(self,input_shape,output_shape,L,lr):
        super().__init__()
        self.l = L
        self.lr = lr
        self.conv1 = nn.Conv1d(4,L,1,padding='valid')
        self.conv2 = nn.Conv1d(L,L,1,padding='valid')
        self.block1 = ResidualBlock(L,11,1)
        self.conv3 = nn.Conv1d(L,L,1,padding='valid')
        self.block2 = ResidualBlock(L,11,4)
        self.conv4 = nn.Conv1d(L,L,1,padding='valid')
        self.block3 = ResidualBlock(L,21,10)
        self.conv5 = nn.Conv1d(L,L,1,padding='valid')
        self.crop_layer = transforms.CenterCrop((L,output_shape[-1]))
        self.output_layer = nn.Conv1d(L,output_shape[0],1,padding='valid')

        self.loss = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.conv2(conv)
        conv = self.block1(conv)
        dense = self.conv3(conv)
        skip = torch.add(skip, dense)
        conv = self.block2(conv)
        dense = self.conv4(conv)
        skip = torch.add(skip, dense)
        conv = self.block3(conv)
        dense = self.conv5(conv)
        skip = torch.add(skip, dense)
        skip = self.crop_layer(skip)
        output = self.output_layer(skip)

        return output

class Splice_AI_80(binary_models):
    def __init__(self,input_shape,output_shape,L,lr):
        super(Splice_AI_80,self).__init__()
        self.l = L
        self.lr = lr
        self.conv1 = nn.Conv1d(4,L,1,padding='valid')
        self.conv2 = nn.Conv1d(L,L,1,padding='valid')
        self.block1 = ResidualBlock(L,11,1)
        self.conv3 = nn.Conv1d(L,L,1,padding='valid')
        self.crop_layer = transforms.CenterCrop((L,output_shape[-1]))
        self.output_layer = nn.Conv1d(L,output_shape[0],1,padding='valid')
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        #metrics

    def forward(self, x):
        conv = self.conv1(x)
        skip = self.conv2(conv)
        conv = self.block1(conv)
        dense = self.conv3(conv)
        skip = torch.add(skip, dense)
        skip = self.crop_layer(skip)
        output = self.output_layer(skip)
        return output

class Residual_annotation(binary_models):
    def __init__(self,input_shape,output_shape,L,lr):
        super().__init__()
        self.lr = lr
        self.bottle_neck=12
        self.length = input_shape[-1]
        self.tar_length = output_shape[-1]
        self.num_task = output_shape[0]
        layers = [
        nn.Conv1d(input_shape[0],L,11,padding='same'),
        nn.BatchNorm1d(L),
        nn.ReLU(),
        dilated_residual(L,L,11,[1,1,1,1],0.1),
        dilated_residual(L,L,8,[4,4,4,4],0.1),
        dilated_residual(L,L,8,[10,10,10,10],0.1),
        nn.Conv1d(L,L,1,padding='same'),
        nn.BatchNorm1d(L),
        nn.ReLU(),
        nn.Dropout(0,1),
        nn.Conv1d(L,self.bottle_neck,1,padding='same'),
        ]
        self.block=nn.Sequential(*layers)

        self.dense = nn.Linear(self.bottle_neck,self.num_task)
        self.crop = transforms.CenterCrop((self.num_task,self.tar_length))

        self.loss_func = nn.CrossEntropyLoss(reduction='sum')

    def forward(self,x):
        conv = self.block(x)
        conv = conv.view(-1,self.length,self.bottle_neck)
        conv = self.dense(conv)
        conv = conv.view(-1,self.num_task,self.length)
        output = self.crop(conv)
        return output

class ResidualUnit(pl.LightningModule):
    def __init__(self,l,w,ar):
        super().__init__()
        block_compose = [
                    nn.BatchNorm1d(l),
                    nn.ReLU(),
                    nn.Conv1d(l,l,w,dilation=ar,padding='same'),
                    nn.BatchNorm1d(l),
                    nn.ReLU(),
                    nn.Conv1d(l,l,w,dilation=ar,padding='same')
                    ]
        self.block = nn.Sequential(*block_compose)

    def forward(self,x):
        output = self.block(x)
        return output

class ResidualBlock(pl.LightningModule):
    def __init__(self,l,w,ar,repeat = 4):
        super().__init__()
        layers=[]
        for i in range(repeat):
            layers.append(ResidualUnit(l,w,ar))
        self.block = nn.Sequential(*layers)
        #self.block.apply(init_weights)
    def forward(self,x):
        output = self.block(x)
        return output

class dilated_residual(pl.LightningModule):
    def __init__(self,filter_num,kernel_size,dilation_rate,dropout):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(filter_num,filter_num,kernel_size,padding='same'))
        layers.append(nn.BatchNorm1d(filter_num))
        for i in range(0,len(dilation_rate)):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(filter_num,filter_num,kernel_size,
                                    padding = 'same',dilation = dilation_rate[i]))
            layers.append(nn.BatchNorm1d(filter_num))
        self.block=nn.Sequential(*layers)
        self.output_act = nn.ReLU()

    def forward(self,x):
        out = self.block(x)
        residual = torch.add(out,x)
        output = self.output_act(residual)
        return output
