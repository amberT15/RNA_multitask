import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from torchinfo import summary
import h5py
import torchmetrics

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
